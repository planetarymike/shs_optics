import copy
import warnings
import numpy as np
import pyvista as pv

import jax
import jax.numpy as jnp
# from jax.tree_util import register_pytree_node_class
jax.config.update('jax_platform_name', 'cpu')
jax.config.update("jax_enable_x64", True)


def add_const_axis(v, axis=-1, r=3):
    # add a new axis at the specified location and duplicate array r
    # times along that axis.
    return jnp.repeat(jnp.expand_dims(v, axis), r, axis=axis)


def const_to_vec3(const):
    # pad an array by repeating 3x at last dimension
    return add_const_axis(const, -1, 3)


def vec_array_dot(vec1, vec2):
    vec1 = jnp.asarray(vec1)
    vec2 = jnp.asarray(vec2)

    # return the dot product of two similar shaped arrays
    return jnp.sum(vec1*vec2, axis=-1)


def vec_array_norm(vec):
    vec = jnp.asarray(vec)
    # return the norm of each vector in an array
    return jnp.sqrt(jnp.sum(vec*vec, axis=-1))


def vec_array_normalize(vec):
    vec = jnp.asarray(vec)
    # return normalized vectors
    norm = vec_array_norm(vec)
    norm = const_to_vec3(norm)
    return vec/norm


def rotation_matrix(angle, vector):

    vector = jnp.array(jnp.copy(vector))
    if vector.shape == (3,):
        vector = jnp.array([vector])
    angle = jnp.array(jnp.copy(angle))
    if angle.shape == ():
        angle = jnp.array([angle])

    vector = vec_array_normalize(vector)
    vx = vector[..., 0]
    vy = vector[..., 1]
    vz = vector[..., 2]

    angle_grid = jnp.meshgrid(angle, vx)[0]
    vx_grid, vy_grid, vz_grid = [jnp.meshgrid(angle, comp)[1]
                                 for comp in [vx, vy, vz]]

    angle = angle_grid
    vx, vy, vz = vx_grid, vy_grid, vz_grid

    cosa = jnp.cos(angle)
    sina = jnp.sin(angle)

    mat = jnp.array([[vx*vx + cosa*(vy*vy + vz*vz),
                      vx*vy*(1.0 - cosa) - vz*sina,
                      vx*vz*(1.0 - cosa) + vy*sina],

                     [vx*vy*(1.0 - cosa) + vz*sina,
                      vy*vy + cosa*(vx*vx + vz*vz),
                      vy*vz*(1.0 - cosa) - vx*sina],

                     [vx*vz*(1.0 - cosa) - vy*sina,
                      vy*vz*(1.0 - cosa) + vx*sina,
                      vz*vz + cosa*(vx*vx + vy*vy)]])
    mat = jnp.moveaxis(mat, -2, 0)
    mat = jnp.moveaxis(mat, -1, 0)

    return jnp.squeeze(mat)


# @register_pytree_node_class
class Rays:
    """generic ray keeper class"""
    def __init__(self,
                 origins,
                 directions,
                 wavelengths=550.,  # nm
                 intensities=1.0,
                 dist_phase=0.0):
        if (((hasattr(origins, 'shape')
              and (origins.shape != directions.shape))
             or len(origins) != len(directions))):
            raise ValueError('size and type'
                             ' of origins and directions must match')

        self.origins = jnp.asarray(origins)
        self.directions = jnp.asarray(directions)
        # make sure directions are normalized
        self.directions = vec_array_normalize(self.directions)

        def add_quantity(name, var):
            if hasattr(var, 'shape') and var.shape != ():
                if var.shape == self.origins.shape[:-1]:
                    setattr(self, name, jnp.asarray(var))
                else:
                    raise ValueError(f'size of {name} and origins must match')
            else:
                arr = jnp.full(self.origins.shape[:-1], var)
                setattr(self, name, arr)

        add_quantity('wavelengths', wavelengths)
        add_quantity('intensities', intensities)

        # dist_phase is the total distance from the origin point of
        # the ray at the start of the optical system, plus any phase
        # difference imposed by diffractive elements encountered by
        # the ray
        add_quantity('dist_phase', dist_phase)

    def __str__(self):
        retstr = f'Ray container: {self.origins.shape[:-1]} rays\n'
        retstr += f'    origins    : {self.origins}\n'
        retstr += f'    directions : {self.directions}\n'
        retstr += f'    wavelengths: {self.wavelengths}\n'
        retstr += f'    intensities: {self.intensities}'
        return retstr

    def __repr__(self):
        return self.__str__()

    @property
    def shape(self):
        return self.origins.shape[:-1]

    def nancopy(self):
        nan_rays = copy.deepcopy(self)
        nan_rays.origins *= jnp.nan
        nan_rays.directions *= jnp.nan
        nan_rays.intensities *= jnp.nan
        nan_rays.dist_phase *= jnp.nan

        return nan_rays

    # def tree_flatten(self):
    #     children = (self.origins,
    #                 self.directions,
    #                 self.wavelengths,
    #                 self.intensities,
    #                 self.dist_phase)
    #     aux_data = None
    #     return (children, aux_data)

    # @classmethod
    # def tree_unflatten(cls, aux_data, children):
    #     return cls(*children)


class Element:
    """Generic optical element"""

    # this class contains default coordinate system and coordinate
    # transformation routines.

    # child classes should define the following methods:
    # * shape (defines surface shape in u, v coordinates)
    # * shape_normal (defines surface normal in element coordinates given u, v)
    # * ray_distance (gets distance between incoming ray
    #                 (in element coordinates) and element,
    #                 may be negative or nan
    # * _ubounds, _vbounds (max extent of element in u, v coords)
    # * in_bounds (defines u, v extent of element)

    # it also contains a default ray intersection method based on
    # pyvista mesh intersections (these are unreliable)
    def __init__(self,
                 center=[0., 0., 0.],
                 normal=[1., 0., 0.],
                 udir=None,
                 vdir=None,
                 reflectivity=1.,
                 transmission=0.,
                 **kwargs):

        self._center = jnp.asarray(center)*1.0

        self._normal = vec_array_normalize(normal)

        self._reflectivity = reflectivity
        self._transmission = transmission

        # figure out element coordinate system
        if ((udir is None) and (vdir is None)):
            # supply a default udir
            if jnp.linalg.norm(jnp.cross(self._normal,
                                         jnp.asarray([1, 0, 0]))) != 0.:
                # if normal is not along x, use closest direction to x
                # for udir
                udir = jnp.array([1, 0, 0])
            else:
                # otherwise, use closest direction to y
                udir = jnp.array([0, 1, 0])

        if (len(center) != 3 or len(normal) != 3
                or (udir is not None and len(udir) != 3)
                or (vdir is not None and len(vdir) != 3)):
            raise ValueError('check dimensions of inputs')

        if udir is not None:
            dir1 = jnp.asarray(udir)*1.0
            uvsign = +1
        else:
            dir1 = jnp.asarray(vdir)*1.0
            uvsign = -1

        # remove any out of plane component and normalize
        dir1 -= self._normal*jnp.dot(dir1, self._normal)
        dir1 = vec_array_normalize(dir1)

        dir2 = jnp.cross(self._normal, dir1)*uvsign

        if udir is not None:
            self._udir = dir1
            self._vdir = dir2
        else:
            self._udir = dir2
            self._vdir = dir1

        self._Rmat = jnp.asarray([self._udir,
                                  self._vdir,
                                  self._normal])
        self._invRmat = jnp.linalg.inv(self._Rmat)

    def get_pvobj(self):
        n_pts = 30
        ucoords = jnp.linspace(self._ubounds[0], self._ubounds[1], n_pts)
        vcoords = jnp.linspace(self._vbounds[0], self._vbounds[1], n_pts)
        uu, vv = jnp.meshgrid(ucoords, vcoords)
        inbounds = self.in_bounds(uu, vv)
        zz = self.shape(uu, vv)
        zz = jnp.where(jnp.logical_not(inbounds), jnp.nan, zz)

        mesh_coords_element = jnp.asarray([uu, vv, zz])
        mesh_coords_element = jnp.moveaxis(mesh_coords_element, 0, -1)

        mesh_coords_universal = \
            self.transform_to_universal(mesh_coords_element)
        mesh_coords_universal = np.copy(mesh_coords_universal)
        mesh_coords_universal = np.moveaxis(mesh_coords_universal, -1, 0)
        pvobj = pv.StructuredGrid(*mesh_coords_universal)

        return pvobj.triangulate()

    @staticmethod
    def check_uv(u, v):
        # check input for shape correctness
        if u.shape != v.shape:
            raise ValueError("Shape of input coordinate arrays must match.")
        return

    def rotate_to_element(self, coords):
        # rotate universal coordinates to element coordinates

        # input coords have any dimensionality, last dimension
        # contains xyz coordinates in universal system

        # output coords are u, v, normal, coords in frame of element
        output = jnp.inner(self._Rmat, coords)

        return jnp.moveaxis(output, 0, -1)

    def shift_to_element(self, coords):
        # shift coordinates so they are centered on the element
        return coords - self._center

    def transform_to_element(self, coords):
        # shift and rotate to centered element coordinates
        return self.rotate_to_element(self.shift_to_element(coords))

    def rotate_to_universal(self, coords):
        # rotate element coordinates to universal coordinates

        # input coords have any dimensionality, last dimension
        # contains xyz coordinates in universal system

        # output coords are u, v, normal, coords in frame of element
        output = jnp.inner(self._invRmat, coords)

        return jnp.moveaxis(output, 0, -1)

    def shift_to_universal(self, coords):
        # shift element coordinates so they are universal
        return coords + self._center

    def transform_to_universal(self, coords):
        # rotate and shift element coordinates to universal
        return self.shift_to_universal(self.rotate_to_universal(coords))

    # def pyvista_intersect(self, rays):
    #     # Default intersection algorithm, based on pyvista mesh
    #     # intersection.

    #     # !! This algorithm frequently fails and should be overriden
    #     # by something more robust in child classes whenever possible

    #     # determine the dimensionality of the input array
    #     raydims = rays.shape
    #     vecdims = raydims+(3,)

    #     # construct flattened arrays for the return values
    #     retpoints = jnp.reshape(jnp.full(vecdims, jnp.nan), (-1, 3))
    #     retnormals = jnp.reshape(jnp.full(vecdims, jnp.nan), (-1, 3))

    #     # find intersections of flattened input rays
    #     flat_origins = jnp.reshape(rays.origins, (-1, 3))
    #     flat_directions = jnp.reshape(rays.directions, (-1, 3))
    #     xpoints, xrays, xcells = self.pvobj.multi_ray_trace(flat_origins,
    #                                                         flat_directions)

    #     # assign to the appropriate location in the flattened return arrays
    #     retpoints[xrays, :] = xpoints
    #     retnormals[xrays, :] = self.pvobj.cell_normals[xcells]

    #     # reshape to the original dimensions of the input rays
    #     retpoints = jnp.reshape(retpoints, vecdims)
    #     retnormals = jnp.reshape(retnormals, vecdims)

    #     # jnp.nan if not intersecting, otherwise 1.
    #     isect = vec_array_norm(retnormals)

    #     return isect, retpoints, retnormals

    def intersect(self, rays):
        # input: Rays
        # output: isect, xpoints, xnormals

        # transform input ray coordinates to element
        elem_origins = self.transform_to_element(rays.origins)
        elem_directions = self.rotate_to_element(rays.directions)

        # get distance between ray origin and element
        elem_dist = self.get_ray_distance(elem_origins, elem_directions)

        # get intersection point
        elem_xpoints = elem_origins + elem_directions*const_to_vec3(elem_dist)

        # check if xpoint is in_bounds
        inbounds = self.in_bounds(elem_xpoints[..., 0], elem_xpoints[..., 1])

        # isect = 1 or jnp.nan based on whether ray intersects
        # intersection occurs if ray is pointed toward plane and
        # intersection point is in bounds
        isect = jnp.where(jnp.logical_and(elem_dist > 0, inbounds),
                          1.0,
                          jnp.nan)

        elem_xpoints = elem_xpoints*const_to_vec3(isect)
        elem_xnormals = self.surf_normal(elem_xpoints[..., 0],
                                         elem_xpoints[..., 1])

        # transform intersections and normals back to universal coordinates
        xpoints = self.transform_to_universal(elem_xpoints)
        xnormals = self.rotate_to_universal(elem_xnormals)

        return isect, xpoints, xnormals

    def get_ray_distance_newton(self, elem_origins, elem_directions):
        # use newton's method and the surface shape, normals to get
        # the ray intersection distance

        # with np.errstate(divide='ignore'):
        dist0 = -elem_origins[..., 2]/elem_directions[..., 2]

        # find intersection distance using Newton's method
        max_newton_iterations = 50
        eps = 1e-8
        error = 1.  # ensure at least one iteration

        it = 0
        dist = dist0
        old_dist = dist

        def cond(val):
            error = val[0]
            it = val[1]
            return ((error > eps) & (it < max_newton_iterations))

        def loop(val):
            error = val[0]
            it = val[1]
            dist = val[2]
            old_dist = val[3]

            # print(f'{it = }')
            coords = elem_origins + const_to_vec3(dist)*elem_directions
            # with jnp.errstate(invalid='ignore', divide='ignore'):
            shape_offset = (coords[..., 2]
                            - self.shape(coords[..., 0], coords[..., 1]))
            normal_vec = self.surf_normal(coords[..., 0], coords[..., 1])
            normal_vec = normal_vec/const_to_vec3(normal_vec[..., 2])
            shape_offset_derivative = jnp.sum(normal_vec, axis=-1)

            old_dist = dist
            update = shape_offset/shape_offset_derivative
            dist += update

            # check for oscillating, growing values
            oscillating = (jnp.sign(dist) != jnp.sign(old_dist))
            larger_magnitude = (jnp.abs(dist) > jnp.abs(old_dist))
            not_intersecting = jnp.logical_and(oscillating,
                                               larger_magnitude)
            dist = jnp.where(not_intersecting, jnp.nan, dist)

            error = jnp.nanmax(jnp.abs(shape_offset))
            it += 1

            return error, it, dist, old_dist

        val = jax.lax.while_loop(cond, loop, (error, it, dist, old_dist))
        dist = val[2]

        # check one last time if error is satisfied
        coords = elem_origins + const_to_vec3(dist)*elem_directions
        shape_offset = (coords[..., 2] - self.shape(coords[..., 0],
                                                    coords[..., 1]))
        not_intersecting = (jnp.abs(shape_offset) >= eps)
        dist = jnp.where(not_intersecting, jnp.nan, dist)

        return dist

    def reflect(self, rays):
        new_rays = rays.nancopy()

        isect, xpoints, xnormals = self.intersect(rays)

        new_rays.origins = xpoints
        ray_normal_dot = vec_array_dot(rays.directions, xnormals)
        ray_normal_dot = const_to_vec3(ray_normal_dot)
        new_rays.directions = rays.directions - 2*ray_normal_dot*xnormals

        # TODO: make a function of wavelength
        new_rays.intensities = rays.intensities*isect

        new_rays.dist_phase = (rays.dist_phase
                               + jnp.linalg.norm(new_rays.origins
                                                 - rays.origins,
                                                 axis=-1))
        return new_rays


class Rectangle(Element):
    """Rectangular optical element, could be detector or mirror"""
    def __init__(self,
                 width=[1., 1.],  # full width in u, v coordinates
                 **kwargs):

        if not hasattr(width, '__len__'):
            width = width * 1.0
            width = jnp.asarray([width, width])

        self._width = width
        self._ubounds = jnp.asarray([-width[0]/2., width[0]/2.])
        self._vbounds = jnp.asarray([-width[1]/2., width[1]/2.])

        # initalize parent last, it needs access to all child methods
        super().__init__(**kwargs)

    def shape(self, u, v):
        # defines surface shape in (u, v) coordinates
        self.check_uv(u, v)

        return jnp.zeros_like(u)

    def in_bounds(self, u, v):
        # returns a boolean array for input u, v coordinates based on
        # whether they are inside the element
        self.check_uv(u, v)

        return jnp.logical_and(jnp.abs(u) <= self._width[0]/2.,
                               jnp.abs(v) <= self._width[1]/2.)

    def surf_normal(self, u, v):
        # returns surface normal vector in element coordinates given
        # (u, v) coordinates
        self.check_uv(u, v)

        return jnp.full(u.shape+(3,), jnp.asarray([0, 0, 1.]))

    def get_ray_distance(self, elem_origins, elem_directions):
        # plane intersections are easy
        dist = -elem_origins[..., 2]/elem_directions[..., 2]
        return dist

    def pass_through(self, rays):
        # pass rays through without modification
        new_rays = rays.nancopy()

        isect, xpoints, _ = self.intersect(rays)

        new_rays.origins = xpoints
        new_rays.directions = rays.directions*const_to_vec3(isect)
        new_rays.intensities = rays.intensities*self._transmission*isect

        new_rays.dist_phase = (rays.dist_phase
                               + jnp.linalg.norm(new_rays.origins
                                                 - rays.origins,
                                                 axis=-1))
        return new_rays

    def detect(self, rays):
        new_rays = rays.nancopy()

        isect, xpoints, _ = self.intersect(rays)

        new_rays.origins = xpoints
        new_rays.directions = rays.directions*const_to_vec3(isect)
        new_rays.intensities = rays.intensities*isect

        new_rays.dist_phase = (rays.dist_phase
                               + jnp.linalg.norm(new_rays.origins
                                                 - rays.origins,
                                                 axis=-1))
        return new_rays

    def detector_coords(self, rays):
        detector_rays = self.detect(rays)

        dcoords = self.transform_to_element(detector_rays.origins)

        return dcoords[..., [0, 1]]


class PlaneGrating(Rectangle):
    """Plane Grating"""
    def __init__(self,
                 grating_vector=None,  # dispersion direction
                 groove_density=None,  # lines / mm
                 **kwargs):
        Rectangle.__init__(self, **kwargs)

        if grating_vector is None:
            warnings.warn('grating_vector not set, defaulting to udir.')
            grating_vector = self._vdir
        grating_vector = jnp.array(grating_vector)
        # remove the along-normal component
        grating_vector = \
            grating_vector - jnp.dot(self._normal, grating_vector)*self._normal
        # normalize
        grating_vector = grating_vector / jnp.linalg.norm(grating_vector)
        self._grating_vector = grating_vector

        if groove_density is None:
            raise ValueError("Please set groove_density (lines/ mm) "
                             "when defining a grating.")
        self._groove_density = groove_density

    def diffract(self, rays, order=1):
        # input, Rays incident on optic
        # output, Rays exiting from optic

        new_rays = rays.nancopy()

        isect, xpoints, xnormals = self.intersect(rays)
        new_rays.origins = xpoints

        # using the vector version of the grating equation given in
        # Spencer&Murty1962, the diffracted ray direction is given by
        # S' = S - disp*grating_vector + gamma*norm,
        #        ^ minus sign here is conventional, it ensures that
        #          the +1 order propagates closest to the incident ray

        # disp = m*λ / d (default λ in nm, 1/d in lines / mm)
        disp = order*rays.wavelengths*1e-6*self._groove_density

        # gamma is whatever it needs to be to make S' a unit vector,
        # and is determined by solving the appropriate quadratic
        # equation.
        c0 = disp*disp - 2*disp*vec_array_dot(rays.directions,
                                              self._grating_vector)
        c1 = vec_array_dot(rays.directions, self._normal)

        discr = c1*c1 - c0
        # the quadratic has two solutions
        # with jnp.errstate(invalid='ignore'):
        g1 = -c1 + jnp.sqrt(discr)
        g2 = -c1 - jnp.sqrt(discr)

        # for reflection, we want the root with larger magnitude
        gamma = jnp.nanmax(jnp.array([g1, g2]), axis=0)

        new_rays.directions = (rays.directions
                               - const_to_vec3(disp)*self._grating_vector
                               + const_to_vec3(gamma)*self._normal)

        # TODO: make a function of wavelength
        new_rays.intensities = rays.intensities*isect

        # update path lengths
        new_rays.dist_phase = (rays.dist_phase
                               + jnp.linalg.norm(new_rays.origins
                                                 - rays.origins,
                                                 axis=-1))

        # account for phase shift introduced by grating
        phase_shift = disp*vec_array_dot(new_rays.origins - self._center,
                                         -self._grating_vector)
        # this minus sign :              ^
        # is a result of selecting S' = S - disp*grating_vector + ...
        # in the output ray calculation above.
        new_rays.dist_phase += phase_shift

        return new_rays


class IdealLens(Element):
    # Ideal Thin Lens, no abberrations
    def __init__(self,
                 width=[1., 1.],  # full width in u, v coordinates
                 f=1.,  # focal length of lens
                 rectangular=True,
                 **kwargs):

        self._rectangular = rectangular
        self._f = f * 1.0

        if not hasattr(width, '__len__'):
            width = jnp.asarray([width, width])

        self._width = width
        self._ubounds = jnp.asarray([-width[0]/2., width[0]/2.])
        self._vbounds = jnp.asarray([-width[1]/2., width[1]/2.])

        if 'reflectivity' not in kwargs.keys():
            kwargs['reflectivity'] = 0.0
        if 'transmission' not in kwargs.keys():
            kwargs['transmission'] = 1.0

        # initalize parent last, it needs access to all child methods
        super().__init__(**kwargs)

    def shape(self, u, v):
        # defines surface shape in (u, v) coordinates
        self.check_uv(u, v)

        return jnp.zeros_like(u)

    def in_bounds(self, u, v):
        # returns a boolean array for input u, v coordinates based on
        # whether they are inside the element
        self.check_uv(u, v)

        if self._rectangular:
            return jnp.logical_and(jnp.abs(u) <= self._width[0]/2.,
                                   jnp.abs(v) <= self._width[1]/2.)

        # else, boundary is elliptical
        return ((u/(self._width[0]/2.))**2
                +
                (v/(self._width[1]/2.))**2
                <= 1.)

    def surf_normal(self, u, v):
        # returns surface normal vector in element coordinates given
        # (u, v) coordinates
        self.check_uv(u, v)

        return jnp.full(u.shape+(3,), jnp.array([0, 0, 1.]))

    def get_ray_distance(self, elem_origins, elem_directions):
        # plane intersections are easy
        return -elem_origins[..., 2]/elem_directions[..., 2]

    def refract(self, rays):
        new_rays = rays.nancopy()

        isect, xpoints, _ = self.intersect(rays)

        new_rays.origins = xpoints

        # off-axis collimated rays are focused in the focal plane
        # a ray passing through the center of the lens is undeflected
        # all other rays must meet that ray at its location in the focal plane
        # this converts sin θ to transverse distance at the focal plane

        # compute the normal component of the ray
        #    jnp.abs here ensures the ray is traveling the same way
        #    wrt the lens normal
        ray_normal_dot = jnp.abs(vec_array_dot(rays.directions,
                                               self._normal))

        # vectors with unit length in the direction of the lens normal
        dirs_unit_length_along_normal = \
            rays.directions/const_to_vec3(ray_normal_dot)

        # determine point where ray passing through center of lens
        # hits the focal plane
        central_ray_vec = dirs_unit_length_along_normal*self._f
        fp_locations = central_ray_vec + self._center

        # determine direction between lens intersection and the above point
        ray_vec = fp_locations - xpoints

        # normalize
        ray_vec_norm = vec_array_normalize(ray_vec)

        new_rays.directions = ray_vec_norm*const_to_vec3(isect)

        # TODO: make wavelength dependent
        new_rays.intensities = rays.intensities*self._transmission*isect

        new_rays.dist_phase = (rays.dist_phase
                               + jnp.linalg.norm(new_rays.origins
                                                 - rays.origins,
                                                 axis=-1))

        # account for phase shift introduced by refraction
        off_center_vec = xpoints - self._center
        off_center_dist2 = jnp.sum(off_center_vec*off_center_vec,
                                   axis=-1)
        extra_ray_dist = off_center_dist2/(2*self._f)
        new_rays.dist_phase -= extra_ray_dist

        return new_rays


class SphericalMirror(Element):
    """Spherical Mirror"""
    def __init__(self,
                 width=[1., 1.],  # full width in u, v coordinates
                 rectangular=True,
                 f=1.,
                 **kwargs):

        self._rectangular = rectangular
        self._f = 1.0*f
        self._sphere_radius = 2.*self._f

        if not hasattr(width, '__len__'):
            width = jnp.asarray([width, width])

        self._width = width
        self._ubounds = jnp.asarray([-width[0]/2., width[0]/2.])
        self._vbounds = jnp.asarray([-width[1]/2., width[1]/2.])

        # initalize parent last, it needs access to all child methods
        super().__init__(**kwargs)

    def shape(self, u, v):
        # defines surface shape in (u, v) coordinates
        self.check_uv(u, v)

        return (self._sphere_radius
                - (jnp.sign(self._sphere_radius)
                   * jnp.sqrt(self._sphere_radius**2
                              - u*u
                              - v*v)))

    def in_bounds(self, u, v):
        # returns a boolean array for input u, v coordinates based on
        # whether they are inside the element
        self.check_uv(u, v)

        if self._rectangular:
            return jnp.logical_and(jnp.abs(u) <= self._width[0]/2.,
                                   jnp.abs(v) <= self._width[1]/2.)

        # else, boundary is elliptical
        return ((u/(self._width[0]/2.))**2
                +
                (v/(self._width[1]/2.))**2
                <= 1.)

    def surf_normal(self, u, v):
        # returns surface normal vector in element coordinates given
        # (u, v) coordinates
        self.check_uv(u, v)

        ucomp = -u
        vcomp = -v
        ncomp = (jnp.sign(self._sphere_radius)
                 * jnp.sqrt(self._sphere_radius**2 - u*u - v*v))

        normals = \
            jnp.asarray([ucomp, vcomp, ncomp])/jnp.abs(self._sphere_radius)
        normals = jnp.moveaxis(normals, 0, -1)

        return normals

    def get_ray_distance(self, elem_origins, elem_directions):
        # sphere intersections are easiest in the sphere-centered frame

        # shift so that sphere center is at origin
        sphere_origins = (elem_origins
                          - jnp.asarray([0, 0, self._sphere_radius]))

        # solve quadratic to obtain distance to sphere
        B = jnp.sum(sphere_origins*elem_directions, axis=-1)
        C = (jnp.sum(sphere_origins*sphere_origins, axis=-1)
             - self._sphere_radius**2)
        discr = B*B - C  # factors of 2 already removed from terms

        # the quadratic has two solutions
        # with jnp.errstate(invalid='ignore'):
        d1 = -B + jnp.sqrt(discr)
        d2 = -B - jnp.sqrt(discr)

        # we need to select the closest point that intersects the
        # correct half of the sphere, so get the along-normal
        # coordinate of both intersection points
        d1_z = elem_origins[..., 2] + d1*elem_directions[..., 2]
        d2_z = elem_origins[..., 2] + d2*elem_directions[..., 2]

        # eliminate incorrect branches
        d1 = jnp.where(jnp.abs(d1_z) > jnp.abs(self._sphere_radius),
                       jnp.nan, d1)
        d2 = jnp.where(jnp.abs(d2_z) > jnp.abs(self._sphere_radius),
                       jnp.nan, d2)

        # select minimum remaining distance (if any)
        dist = jnp.nanmin(jnp.array([d1, d2]), axis=0)

        return dist


def quartic_roots(p):
    # finds roots of quartic equations by finding the eigenvalues of
    # the companion matrix
    # input: numpy ndarray of shape (any, 5)
    # output: roots of all input equations, shape (any, 4)
    #         roots may be complex!

    # convert coefs to normal form
    dcba = jnp.array([p[..., 4]/p[..., 0],
                      p[..., 3]/p[..., 0],
                      p[..., 2]/p[..., 0],
                      p[..., 1]/p[..., 0]])
    dcba = -jnp.moveaxis(dcba, 0, -1)

    # construct the companion matrix
    companion_matrix = jnp.zeros(dcba.shape[:-1] + (4, 4))
    companion_matrix = companion_matrix.at[..., 1:, :3].set(jnp.eye(3))
    companion_matrix = companion_matrix.at[..., :, 3].set(dcba)

    # find roots of non-nan eqns
    nancoefs = jnp.any(jnp.isnan(companion_matrix), axis=-1)
    nancoefs_mat = jnp.repeat(nancoefs[..., jnp.newaxis], 4, axis=-1)

    companion_matrix = jnp.where(nancoefs_mat, 0.0, companion_matrix)
    roots = jnp.linalg.eigvals(companion_matrix)
    roots = jnp.where(nancoefs, jnp.nan, roots)

    # return sorted roots for each input set of coefs
    return jnp.sort(roots, axis=-1)


class ToroidalMirror(Element):
    """Toroidal Mirror"""
    def __init__(self,
                 width=[1., 1.],  # full width in u, v coordinates
                 rectangular=True,
                 fu=1.,  # focal length in u, v
                 fv=1.,
                 **kwargs):

        self._rectangular = rectangular
        fu = jnp.abs(fu)
        fv = jnp.abs(fv)

        # if fu < 0 or fv < 0:
        #     raise ValueError("fu and fv must be > 0 in ToroidalMirror")
        # if fu == fv:
        #     self._torus_is_sphere = True
        #     self._spherical_mirror = SphericalMirror(width=width,
        #                                              rectangular=rectangular,
        #                                              f=fu,
        #                                              **kwargs)
        # else:
        #     self._torus_is_sphere = False

        self._fu = 1.0*fu
        self._fv = 1.0*fv

        # determine small and big focal length
        self._fs = jnp.min(jnp.array([self._fu, self._fv]))
        self._fb = jnp.max(jnp.array([self._fu, self._fv]))

        self._swap_uv = jax.lax.cond(self._fs == self._fv,
                                     lambda x: False,
                                     lambda x: True,
                                     1)

        # if self._fs == self._fv:
        #     self._swap_uv = False
        # else:
        #     self._swap_uv = True

        self._radius_s = 2*self._fs
        self._radius_b = 2*(self._fb - self._fs)

        if not hasattr(width, '__len__'):
            width = jnp.asarray([width, width])

        self._width = width
        self._ubounds = jnp.asarray([-width[0]/2., width[0]/2.])
        self._vbounds = jnp.asarray([-width[1]/2., width[1]/2.])

        # if self._width[1]/2. >= self._radius_s:
        #     raise ValueError("v half width > radius")

        # initalize parent last, it needs access to all child methods
        super().__init__(**kwargs)

    def shape(self, u, v):
        # defines surface shape in (u, v) coordinates
        self.check_uv(u, v)

        s, b = jax.lax.cond(self._swap_uv,
                            lambda x: x,
                            lambda x: (x[1], x[0]),
                            (u, v))

        # with jnp.errstate(invalid='ignore'):
        return (self._radius_b + self._radius_s
                - jnp.sqrt((self._radius_b
                            + jnp.sqrt(self._radius_s*self._radius_s - s*s))**2
                           - b*b))

    def in_bounds(self, u, v):
        # returns a boolean array for input u, v coordinates based on
        # whether they are inside the element
        self.check_uv(u, v)

        if self._rectangular:
            return jnp.logical_and(jnp.abs(u) <= self._width[0]/2.,
                                   jnp.abs(v) <= self._width[1]/2.)

        # else, boundary is elliptical
        return ((u/(self._width[0]/2.))**2
                +
                (v/(self._width[1]/2.))**2
                <= 1.)

    def surf_normal(self, u, v):
        # returns surface normal vector in element coordinates given
        # (u, v) coordinates
        self.check_uv(u, v)

        # convert u, v to torus coordinates
        s, b = jax.lax.cond(self._swap_uv,
                            lambda x: x,
                            lambda x: (x[1], x[0]),
                            (u, v))

        # not normalized yet
        bcomp = -b
        scomp = -s*(1.+self._radius_b/jnp.sqrt(self._radius_s**2 - s*s))
        ncomp = self._radius_b + self._radius_s - self.shape(u, v)

        normals = jnp.asarray([bcomp, scomp, ncomp])
        normals = jnp.moveaxis(normals, 0, -1)

        # normalize
        normnorm = jnp.linalg.norm(normals, axis=-1)
        normals = normals / const_to_vec3(normnorm)

        # convert back to u, v coordinates if needed
        normals = jax.lax.cond(self._swap_uv,
                               lambda x: x[..., [1, 0, 2]],
                               lambda x: x,
                               normals)

        return normals

    def get_ray_distance(self, elem_origins, elem_directions):
        return self.get_ray_distance_newton(elem_origins, elem_directions)

    def get_ray_distance_quartic(self, elem_origins, elem_directions):
        # torus intersections are easiest in the torus-centered frame

        # if self._torus_is_sphere:
        #     return self._spherical_mirror.get_ray_distance(elem_origins,
        #                                                    elem_directions)

        # shift so that torus center is at origin
        def rotate_to_torus(p):
            tp = jnp.copy(p)
            tp = tp.at[..., 2].multiply(-1)

            tp = jax.lax.cond(self._swap_uv,
                              lambda x: x[..., [1, 0, 2]],
                              lambda x: x,
                              tp)

            return tp

        def transform_to_torus(p):
            tp = rotate_to_torus(p)
            tp = tp.at[..., 2].add(self._radius_b + self._radius_s)
            return tp

        torus_origins = transform_to_torus(elem_origins)
        torus_directions = rotate_to_torus(elem_directions)

        # we need to solve a fourth-degree polynomial to get the
        # ray-torus intersections

        # build the polynomial coefficients
        p2 = jnp.sum(torus_origins*torus_origins, axis=-1)
        p_s = torus_origins[..., 1]
        ls = torus_directions[..., 1]
        pdotl = jnp.sum(torus_origins*torus_directions, axis=-1)
        pls = torus_origins[..., 1]*torus_directions[..., 1]
        Rb2 = self._radius_b*self._radius_b
        Rs2 = self._radius_s*self._radius_s

        # I derived these polynomial coefficients in Mathematica
        c0 = p2*p2 + 4*p_s*p_s*Rb2 + (Rb2 - Rs2)**2 - 2*p2*(Rb2 + Rs2)
        c1 = 4*p2*pdotl + 8*pls*Rb2 - 4*pdotl*(Rb2 + Rs2)
        c2 = 2*(p2 + 2*pdotl*pdotl - Rb2 + 2*ls*ls*Rb2 - Rs2)
        c3 = 4*pdotl
        c4 = jnp.ones_like(pdotl)

        # solve the quartic polynomial for intersection distances
        coefs = jnp.moveaxis(jnp.array([c4, c3, c2, c1, c0]), 0, -1)
        roots = quartic_roots(coefs)

        # reject imaginary solutions
        def reject_imag(q, im_thresh=1e-6):
            # input: arbitrary ndarray of imaginary numbers
            # output: real component if number is within angle
            #         im_thresh of real line, otherwise jnp.nan

            pos_im_angle = jnp.abs(jnp.angle(q))
            is_pos_real = jnp.logical_and(q.real > 0,
                                          pos_im_angle < im_thresh)

            neg_im_angle = jnp.abs(jnp.abs(jnp.angle(q)) - jnp.pi)
            is_neg_real = jnp.logical_and(q.real < 0,
                                          neg_im_angle < im_thresh)
            is_real = jnp.logical_or(is_pos_real, is_neg_real)

            retval = jnp.where(is_real, q.real, jnp.nan)

            return retval

        dist = reject_imag(roots)

        # eliminate incorrect branches
        points = (add_const_axis(elem_origins, -2, 4)
                  + (const_to_vec3(dist)
                     * add_const_axis(elem_directions, -2, 4)))

        def is_incorrect_branch(points, atol=1e-6, rtol=1e-3):
            # input: ndarray shape (any, 3) --- point coordinates
            # output: bool ndarray shape (any).
            #         True if point is not on surface

            testcoord = self.shape(points[..., 0], points[..., 1])
            on_surface = jnp.isclose(testcoord, points[..., 2],
                                     rtol=rtol, atol=atol)
            return jnp.logical_not(on_surface)

        dist = jnp.where(is_incorrect_branch(points), jnp.nan, dist)

        # select minimum remaining distance (if any)
        # with warnings.catch_warnings():
        #     warnings.simplefilter("ignore", category=RuntimeWarning)
        dist = jnp.nanmin(dist, axis=-1)

        return dist


class ParabolicMirror(Element):
    """Parabolic Mirror"""
    def __init__(self,
                 width=[1., 1.],  # full width in u, v coordinates
                 rectangular=True,
                 f=1.,
                 **kwargs):

        self._rectangular = rectangular
        self._f = 1.0*f
        self._const = 0.25/self._f

        if not hasattr(width, '__len__'):
            width = jnp.asarray([width, width])

        self._width = width
        self._ubounds = jnp.asarray([-width[0]/2., width[0]/2.])
        self._vbounds = jnp.asarray([-width[1]/2., width[1]/2.])

        # initalize parent last, it needs access to all child methods
        super().__init__(**kwargs)

    def shape(self, u, v):
        # defines surface shape in (u, v) coordinates
        self.check_uv(u, v)

        r2 = u*u + v*v
        surf = self._const*r2

        return surf

    def in_bounds(self, u, v):
        # returns a boolean array for input u, v coordinates based on
        # whether they are inside the element
        self.check_uv(u, v)

        if self._rectangular:
            return jnp.logical_and(jnp.abs(u) <= self._width[0]/2.,
                                   jnp.abs(v) <= self._width[1]/2.)

        # else, boundary is elliptical
        return ((u/(self._width[0]/2.))**2
                +
                (v/(self._width[1]/2.))**2
                <= 1.)

    def surf_normal(self, u, v):
        # returns surface normal vector in element coordinates given
        # (u, v) coordinates
        self.check_uv(u, v)

        ucomp = -2 * self._const * u
        vcomp = -2 * self._const * v
        ncomp = jnp.ones_like(u)

        normals = jnp.asarray([ucomp, vcomp, ncomp])
        normals = jnp.moveaxis(normals, 0, -1)
        normals = vec_array_normalize(normals)

        return normals

    def get_ray_distance(self, elem_origins, elem_directions):
        return self.get_ray_distance_newton(elem_origins,
                                            elem_directions)


class ParabolicCylinder(Element):
    """Parabolic Cylinder. Parabolic Shape is only in u coordinate"""
    def __init__(self,
                 width=[1., 1.],  # full width in u, v coordinates
                 rectangular=True,
                 f=1.,
                 axis_offset=0.,
                 **kwargs):

        self._rectangular = rectangular
        self._f = 1.0*f
        self._const = 0.25/self._f
        self._axis_offset = 1.0*axis_offset
        self._s0 = self._const * self._axis_offset * self._axis_offset

        if not hasattr(width, '__len__'):
            width = jnp.asarray([width, width])

        self._width = width
        self._ubounds = jnp.asarray([-width[0]/2., width[0]/2.])
        self._vbounds = jnp.asarray([-width[1]/2., width[1]/2.])

        # initalize parent last, it needs access to all child methods
        super().__init__(**kwargs)

    def shape(self, u, v):
        # defines surface shape in (u, v) coordinates
        self.check_uv(u, v)

        uo = u - self._axis_offset
        r2 = uo*uo
        surf = self._const*r2 - self._s0

        return surf

    def in_bounds(self, u, v):
        # returns a boolean array for input u, v coordinates based on
        # whether they are inside the element
        self.check_uv(u, v)

        if self._rectangular:
            return jnp.logical_and(jnp.abs(u) <= self._width[0]/2.,
                                   jnp.abs(v) <= self._width[1]/2.)

        # else, boundary is elliptical
        return ((u/(self._width[0]/2.))**2
                +
                (v/(self._width[1]/2.))**2
                <= 1.)

    def surf_normal(self, u, v):
        # returns surface normal vector in element coordinates given
        # (u, v) coordinates
        self.check_uv(u, v)

        uo = u - self._axis_offset
        ucomp = -2 * self._const * uo
        vcomp = jnp.zeros_like(u)
        ncomp = jnp.ones_like(u)

        normals = jnp.asarray([ucomp, vcomp, ncomp])
        normals = jnp.moveaxis(normals, 0, -1)
        normals = vec_array_normalize(normals)

        return normals

    def get_ray_distance(self, elem_origins, elem_directions):
        return self.get_ray_distance_newton(elem_origins,
                                            elem_directions)


class ConicMirror(Element):
    """Conic Mirror, Spencer and Murty 1962"""
    def __init__(self,
                 width=[1., 1.],  # full width in u, v coordinates
                 rectangular=True,
                 c=1.,
                 k=0.,
                 alpha=[0.],
                 **kwargs):

        self._rectangular = rectangular
        self._c = 1.0*c
        self._k = 1.0*k
        self._alpha = jnp.array(alpha)

        if not hasattr(width, '__len__'):
            width = jnp.asarray([width, width])

        self._width = width
        self._ubounds = jnp.asarray([-width[0]/2., width[0]/2.])
        self._vbounds = jnp.asarray([-width[1]/2., width[1]/2.])

        # initalize parent last, it needs access to all child methods
        super().__init__(**kwargs)

    def shape(self, u, v):
        # defines surface shape in (u, v) coordinates
        self.check_uv(u, v)

        r2 = u*u + v*v
        surf = self._c*r2/(1+jnp.sqrt(1-self._k*self._c*self._c*r2))

        rpow = jnp.repeat(jnp.expand_dims(r2, 0),
                          len(self._alpha),
                          axis=0)
        rpow = jnp.cumprod(rpow, axis=0)
        rpow = jnp.moveaxis(rpow, 0, -1)

        surf = surf + jnp.dot(rpow, self._alpha)

        return surf

    def in_bounds(self, u, v):
        # returns a boolean array for input u, v coordinates based on
        # whether they are inside the element
        self.check_uv(u, v)

        if self._rectangular:
            return jnp.logical_and(jnp.abs(u) <= self._width[0]/2.,
                                   jnp.abs(v) <= self._width[1]/2.)

        # else, boundary is elliptical
        return ((u/(self._width[0]/2.))**2
                +
                (v/(self._width[1]/2.))**2
                <= 1.)

    def surf_normal(self, u, v):
        # returns surface normal vector in element coordinates given
        # (u, v) coordinates
        self.check_uv(u, v)

        r2 = u*u + v*v
        coef = self._c/jnp.sqrt(1-self._k*self._c*self._c*r2)

        rpow = jnp.repeat(jnp.expand_dims(r2, 0),
                          len(self._alpha),
                          axis=0)
        rpow = rpow.at[0, ...].set(1.0)
        rpow = jnp.cumprod(rpow, axis=0)
        rpow = jnp.moveaxis(rpow, 0, -1)

        print(f'{rpow.shape = }')
        print(f'{self._alpha = }')
        print(f'{(jnp.arange(len(self._alpha))+1)}')
        alphaj = self._alpha * (jnp.arange(len(self._alpha))+1)
        coef = coef + 2*jnp.dot(rpow, alphaj)

        ucomp = -coef * u
        vcomp = -coef * v
        ncomp = jnp.ones_like(u)

        normals = jnp.asarray([ucomp, vcomp, ncomp])
        normals = jnp.moveaxis(normals, 0, -1)
        normals = vec_array_normalize(normals)

        return normals

    def get_ray_distance(self, elem_origins, elem_directions):
        return self.get_ray_distance_newton(elem_origins,
                                            elem_directions)


class OffAxisParabola(Element):
    """Parabolic Mirror"""
    def __init__(self,
                 width=[1., 1.],  # full width in u, v coordinates,
                                  # measured from center
                 rectangular=True,
                 center=[0., 0., 0.],  # center of off-axis segment,
                                       # NOT vertex of parabola
                 axis_out=[1., 0., 0.],  # direction of OUTGOING
                                         # collimated light
                 focus=[1., 0., 0.],  # 3D location of focus in
                                      # universal coordinates
                 **kwargs):
        # We define an off-axis parabola using 3D universal
        # coordinates for the location of the mirror center and
        # focus.

        # Find the location of the parabola vertex. A parabola has the
        # property that the distance from the focus to any point on
        # the curve is the same as the distance from that point to the
        # latus rectum, which is a horizontal line the same distance
        # from the vertex as the focus, on the opposite side of the
        # vertex.

        # get the distance between the center of the mirror and the
        # focus
        self._center = jnp.asarray(center)
        self._focus = jnp.asarray(focus)
        center_focus_dist = vec_array_norm(self._center - self._focus)

        # this distance is not the same as the distance between the
        # focus and the latus rectum, but we can make it the same by
        # subtracting the along-normal component of the focus ->
        # center vector
        self._axis_out = vec_array_normalize(jnp.asarray(axis_out))
        self._normal = jnp.asarray(axis_out)
        center_focus_along_normal_dist = jnp.dot(self._center - self._focus,
                                                 self._normal)
        focus_lr_dist = center_focus_dist - center_focus_along_normal_dist
        focus_vertex_dist = focus_lr_dist / 2.0

        self._vertex = self._focus - focus_vertex_dist*self._normal
        self._quadratic_coef = 0.25/focus_vertex_dist

        self._rectangular = rectangular
        if not hasattr(width, '__len__'):
            width = jnp.asarray([width, width])

        self._width = width
        self._ubounds = jnp.asarray([-width[0]/2., width[0]/2.])
        self._vbounds = jnp.asarray([-width[1]/2., width[1]/2.])

        # initalize parent, coordinate transformations
        super().__init__(**kwargs,
                         center=self._center,
                         normal=self._normal)

        # determine (u, v) offset of center from vertex
        self._u0, self._v0, self._s0 = self.transform_to_element(self._vertex)

    def transform_to_parabola(self, u, v):
        return (u - self._u0,  v - self._v0)

    def shape(self, u, v):
        # defines surface shape in (u, v) coordinates
        self.check_uv(u, v)

        pu, pv = self.transform_to_parabola(u, v)

        r2 = pu*pu + pv*pv
        surf = self._quadratic_coef*r2 + self._s0

        return surf

    def in_bounds(self, u, v):
        # returns a boolean array for input u, v coordinates based on
        # whether they are inside the element
        self.check_uv(u, v)

        if self._rectangular:
            return jnp.logical_and(jnp.abs(u) <= self._width[0]/2.,
                                   jnp.abs(v) <= self._width[1]/2.)

        # else, boundary is elliptical
        return ((u/(self._width[0]/2.))**2
                +
                (v/(self._width[1]/2.))**2
                <= 1.)

    def surf_normal(self, u, v):
        # returns surface normal vector in element coordinates given
        # (u, v) coordinates
        self.check_uv(u, v)

        pu, pv = self.transform_to_parabola(u, v)

        ucomp = -2 * self._quadratic_coef * pu
        vcomp = -2 * self._quadratic_coef * pv
        ncomp = jnp.ones_like(u)

        normals = jnp.asarray([ucomp, vcomp, ncomp])
        normals = jnp.moveaxis(normals, 0, -1)
        normals = vec_array_normalize(normals)

        return normals

    def get_ray_distance(self, elem_origins, elem_directions):
        return self.get_ray_distance_newton(elem_origins,
                                            elem_directions)

# class OffAxisParabolicCylinder(OffAxisParabola):
#     """Off Axis Parabolic Cylinder Mirror"""
#     def __init__(self,
#                  width=[1., 1.],  # full width in u, v coordinates,
#                                   # measured from center
#                  rectangular=True,
#                  center=[0., 0., 0.],  # center of off-axis segment,
#                                        # NOT vertex of parabola
#                  axis_out=[1., 0., 0.],  # direction of OUTGOING
#                                          # collimated light
#                  focus_u=[1., 0., 0.],  # 3D location of focus in
#                                         # universal coordinates
#                  **kwargs):

#         self._focus_u = jnp.array(focus_u)

#         # we can use the architecture of the full off-axis paraboloid
#         # and just eliminate the curvature along the v axis
#         super().__init__(width=width,
#                          rectangular=rectangular,
#                          center=center,
#                          axis_out=axis_out,
#                          focus=focus_u,
#                          **kwargs)

#     def shape(self, u, v):
#         # defines surface shape in (u, v) coordinates
#         self.check_uv(u, v)

#         pu, pv = self.transform_to_parabola(u, v)

#         # ignore variation along the v axis because this is a
#         # parabolic cylinder with curvature in the u direction only
#         r2 = pu*pu  # + pv*pv
#         surf = self._quadratic_coef*r2 + self._s0

#         return surf

#     def surf_normal(self, u, v):
#         # returns surface normal vector in element coordinates given
#         # (u, v) coordinates
#         self.check_uv(u, v)

#         paraboloid_normals = super().surf_normal(u, v)

#         # zero out the v component and renormalize
#         normals = paraboloid_normals
#         normals.at[..., 1].set(0.)
#         normals = vec_array_normalize(normals)

#         return normals


class OffAxisBiParabola(Element):
    """Parabolic Mirror"""
    def __init__(self,
                 width=[1., 1.],  # full width in u, v coordinates,
                                  # measured from center
                 rectangular=True,
                 center=[0., 0., 0.],  # center of off-axis segment,
                                       # NOT vertex of parabola
                 axis_out=[1., 0., 0.],  # direction of OUTGOING
                                         # collimated light
                 focus_u=[1., 0., 0.],  # 3D location of u-axis focal
                                        # point in universal
                                        # coordinates
                 focus_v=[1., 0., 0.],  # 3D location of v-axis focal
                                        # point in universal
                                        # coordinates
                 **kwargs):
        # The off-axis biparabola is defined using an off axis
        # parabola in each dimension to define the shape of the
        # surface in that direction.

        self._center = jnp.asarray(center)
        self._axis_out = vec_array_normalize(jnp.asarray(axis_out))
        self._normal = jnp.asarray(axis_out)

        self._rectangular = rectangular
        if not hasattr(width, '__len__'):
            width = jnp.asarray([width, width])

        self._width = width
        self._ubounds = jnp.asarray([-width[0]/2., width[0]/2.])
        self._vbounds = jnp.asarray([-width[1]/2., width[1]/2.])

        # initalize parent, coordinate transformations
        super().__init__(**kwargs,
                         center=self._center,
                         normal=self._normal)

        self._u_parabola = OffAxisParabola(width=width,
                                           rectangular=rectangular,
                                           center=center,
                                           axis_out=axis_out,
                                           focus=focus_u,
                                           **kwargs)
        self._v_parabola = OffAxisParabola(width=width,
                                           rectangular=rectangular,
                                           center=center,
                                           axis_out=axis_out,
                                           focus=focus_v,
                                           **kwargs)

    def shape(self, u, v):
        # defines surface shape in (u, v) coordinates
        self.check_uv(u, v)

        ucomp = self._u_parabola.shape(u, jnp.zeros_like(v))
        vcomp = self._v_parabola.shape(jnp.zeros_like(u), v)
        surf = ucomp + vcomp

        return surf

    def in_bounds(self, u, v):
        # returns a boolean array for input u, v coordinates based on
        # whether they are inside the element
        self.check_uv(u, v)

        if self._rectangular:
            return jnp.logical_and(jnp.abs(u) <= self._width[0]/2.,
                                   jnp.abs(v) <= self._width[1]/2.)

        # else, boundary is elliptical
        return ((u/(self._width[0]/2.))**2
                +
                (v/(self._width[1]/2.))**2
                <= 1.)

    def surf_normal(self, u, v):
        # returns surface normal vector in element coordinates given
        # (u, v) coordinates
        self.check_uv(u, v)

        unorm = self._u_parabola.surf_normal(u, jnp.zeros_like(v))
        vnorm = self._v_parabola.surf_normal(jnp.zeros_like(u), v)

        # we must undo the normalization imposed by the surf_normal
        # function for each parabola component and normalize again
        ucomp = unorm[..., 0]/unorm[..., 2]
        vcomp = vnorm[..., 1]/vnorm[..., 2]
        ncomp = jnp.ones_like(u)

        normals = jnp.asarray([ucomp, vcomp, ncomp])
        normals = jnp.moveaxis(normals, 0, -1)
        normals = vec_array_normalize(normals)

        return normals

    def get_ray_distance(self, elem_origins, elem_directions):
        return self.get_ray_distance_newton(elem_origins,
                                            elem_directions)
