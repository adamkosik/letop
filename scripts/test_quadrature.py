import firedrake as fd
import numpy as np
from finat.point_set import PointSet
from finat.quadrature import QuadratureRule, make_quadrature
from firedrake import READ, WRITE, assemble, dx, par_loop
from ufl.core.expr import Expr


def integral_level_set(f: Expr, ls: fd.Function, dx: fd.Measure) -> fd.Form:
    """ This function modifies the quadrature points in the elements sliced by
    the level-set function. It returns a Form as a sum of integrals.

    Args:
        f (Expr): Integrand.
        ls (Function): The level-set function. It should be CG(1) function.
        dx (Measure):  Integral measure.

    Returns:
        Form: A firedrake form will be returned, which can then be assembled.

    """
    mesh = ls.function_space().mesh()

    quad_n = fd.Constant((3.0))
    quad_weights_ref = fd.Constant((1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0))
    quad_x_ref = fd.Constant((1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0))
    quad_y_ref = fd.Constant((1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0))

    space_indicator = fd.FunctionSpace(mesh, "DG", 0)
    space_quadrature = fd.VectorFunctionSpace(mesh, "DG", 0, dim=9)

    quad_indicator = fd.Function(space_indicator)
    quad_indicator.assign(0.0)
    quad_x = fd.Function(space_quadrature)
    quad_x.assign(0.0)
    quad_y = fd.Function(space_quadrature)
    quad_y.assign(0.0)
    quad_weights = fd.Function(space_quadrature)
    quad_weights.assign(0.0)

    par_loop(
        """
            // Assumption: the reference element has coordinates
            // (0,0), (1,0), (0,1)
            // positively oriented but not necessarily ordered
            double min_value = 1e20;
            double max_value = -1e20;

            int node_flag[3] = {1, 1, 1};	//
            int element_flag = 0;			//

            int n_q = n_q_ref[0];

            // Flag every nodes and gives a binary representation of the level set function
            // 0,7 -> nothing happens (all negative reps. all positive)
            // 1,2,4 -> First, second resp. third node are positive
            // 3,5,6 -> Third, second resp. first node are negative

            for (int i = 0; i < 3; i++) {

                min_value = fmin(0, ls[i]);

                if (min_value < 0) {
                    node_flag[i] = 0;
                }

                element_flag += node_flag[i]* pow(2, i);

            }

            // To compute where the level set crosses the reference element the following cases are equivalent
            // 1 and 6, 2 and 5, 3 and 4
            double crossing_point_x[2] = { 0.0 };
            double crossing_point_y[2] = { 0.0 };

            double triangle_x[3][3] = {{ 0.0 }};
            double triangle_y[3][3] = {{ 0.0 }};
            double det_triangle[3] = { 0.0 };
            double B_triangle[3][2][2] = {{{ 0.0 }}};
            double b_triangle[3][2] = {{ 0.0 }};
            // Transformed quadrature rule
            double w_q_triangle[3][n_q];
            memset(w_q_triangle, 0, 3*n_q*sizeof(double));
            double x_q_triangle[3][n_q];
            memset(x_q_triangle, 0, 3*n_q*sizeof(double));
            double y_q_triangle[3][n_q];
            memset(y_q_triangle, 0, 3*n_q*sizeof(double));

            if (element_flag == 1 || element_flag == 6) {

                crossing_point_x[0] = -ls[0] / (ls[1] - ls[0]); // ls[1] and ls[0] are different because of the different sign
                crossing_point_y[1] = -ls[0] / (ls[2] - ls[0]); // ls[2] and ls[0] are different because of the different sign

                // Transformation for the first triangle
                triangle_x[0][1] = crossing_point_x[0];
                triangle_y[0][2] = crossing_point_y[1];

                // Transformation for the second triangle
                triangle_x[1][0] = crossing_point_x[0];
                triangle_y[1][1] = crossing_point_y[1];
                triangle_x[1][2] = 1.0;

                triangle_x[2][1] = 1.0;
                triangle_y[2][0] = crossing_point_y[1];
                triangle_y[2][2] = 1.0;

            }

            if (element_flag == 2 || element_flag == 5) {

                crossing_point_x[0] = -ls[0] / (ls[1] - ls[0]);	// See above
                crossing_point_x[1] = -ls[2] / (ls[1]-ls[2]);
                crossing_point_y[1] = ls[1] / (ls[1] - ls[2]);

                triangle_x[0][0] = crossing_point_x[0];
                triangle_x[0][1] = 1.0;
                triangle_x[0][2] = crossing_point_x[1];
                triangle_y[0][2] = crossing_point_y[1];

                triangle_x[1][1] = crossing_point_x[0];
                triangle_x[1][2] = crossing_point_x[1];
                triangle_y[1][2] = crossing_point_y[1];

                triangle_x[2][1] = crossing_point_x[1];
                triangle_y[2][1] = crossing_point_y[1];
                triangle_y[2][2] = 1.0;

            }

            if (element_flag == 3 || element_flag == 4) {

                crossing_point_x[0] = -ls[2] / (ls[1] - ls[2]);	// See above
                crossing_point_y[0] = ls[1] / (ls[1] - ls[2]);
                crossing_point_y[1] = -ls[0] / (ls[2] - ls[0]);

                triangle_y[0][0] = crossing_point_y[1];
                triangle_x[0][1] = crossing_point_x[0];
                triangle_y[0][1] = crossing_point_y[0];
                triangle_y[0][2] = 1;

                triangle_x[1][1] = crossing_point_x[0];
                triangle_y[1][1] = crossing_point_y[0];
                triangle_y[1][2] = crossing_point_y[1];

                triangle_x[2][1] = 1;
                triangle_x[2][2] = crossing_point_x[0];
                triangle_y[2][2] = crossing_point_y[0];

            }

            for (int j; j < 3; j++) {
                det_triangle[j] = fabs((triangle_x[j][1] - triangle_x[j][0]) * (triangle_y[j][2] - triangle_y[j][0]) - (triangle_x[j][2] - triangle_x[j][0]) * (triangle_y[j][1] - triangle_y[j][0]));

                // Matrix and vector for transformation to the first triangle
                b_triangle[j][0] = triangle_x[j][0];
                b_triangle[j][1] = triangle_y[j][0];
                B_triangle[j][0][0] = triangle_x[j][1] - triangle_x[j][0];
                B_triangle[j][1][0] = triangle_y[j][1] - triangle_y[j][0];
                B_triangle[j][0][1] = triangle_x[j][2] - triangle_x[j][0];
                B_triangle[j][1][1] = triangle_y[j][2] - triangle_y[j][0];

                // Tranformation of the quadrature rule to the subtriangles
                for (int k = 0; k < n_q; k++) {
                    w_q_triangle[j][k] = det_triangle[j] * w_q_ref[k];
                    x_q_triangle[j][k] = B_triangle[j][0][0] * x_q_ref[k] + B_triangle[j][0][1] * y_q_ref[k] + b_triangle[j][0];
                    y_q_triangle[j][k] = B_triangle[j][1][0] * x_q_ref[k] + B_triangle[j][1][1] * y_q_ref[k] + b_triangle[j][1];
                }

            }

            for (int i=0; i<ls.dofs; i++) {
                min_value = fmin(min_value, ls[i]);
                max_value = fmax(max_value, ls[i]);
            }
            if (min_value < 0 && max_value > 0) {
                q_ind[0] = 1.0;
            }
            else {
                q_ind[0] = 0.0;
            }
            for (int j = 0; j < 3; j++) {
                for (int k = 0; k < n_q; k++) {
                    w[n_q*j+k] = w_q_triangle[j][k];
                    x[n_q*j+k] = x_q_triangle[j][k];
                    y[n_q*j+k] = y_q_triangle[j][k];
                }
            }
        """,
        dx,
        {
            'ls': (ls, READ),
            'n_q_ref': (quad_n, READ),
            'w_q_ref': (quad_weights_ref, READ),
            'x_q_ref': (quad_x_ref, READ),
            'y_q_ref': (quad_y_ref, READ),
            'q_ind': (quad_indicator, WRITE),
            'x': (quad_x, WRITE),
            'y': (quad_y, WRITE),
            'w': (quad_weights, WRITE),
        }
    )

    # First, we prepare an integral form for all elements not sliced by the
    # level-set function.
    inv_indicator = fd.Function(space_indicator)
    inv_indicator.assign(1.0)
    for idx, indicator in enumerate(quad_indicator.dat.data):
        if indicator == 1.0:
            inv_indicator.dat.data[idx] = 0.0
    form = f * inv_indicator * dx

    # In the next step, we add all integral forms for all elements sliced by
    # the level-set function. Each cell has a single integral form.
    for idx, indicator in enumerate(quad_indicator.dat.data):
        if indicator == 1.0:
            marker = fd.Function(space_indicator)
            marker.assign(0.0)
            marker.dat.data[idx] = 1.0

            points1 = PointSet(
                np.stack((quad_x.dat.data[idx], quad_y.dat.data[idx]), axis=-1)
            )
            weights1 = quad_weights.dat.data[idx]
            rule2D = QuadratureRule(points1, weights1)

            form += f * marker * dx(scheme=rule2D)

    return form


# Test functions:
def test_function_integration() -> None:
    mesh = fd.UnitSquareMesh(2, 2)
    S = fd.FunctionSpace(mesh, "CG", 1)
    f = fd.Function(S)
    ls = fd.Function(S)

    x, y = fd.SpatialCoordinate(mesh)
    f.interpolate(x)
    ls.interpolate(x + y - 1.31)

    integral = integral_level_set(f, ls, dx)

    print(assemble(integral))

    print(assemble(f * dx))


def heaviside(f, x, y) -> fd.Function:
    return fd.conditional(
        fd.lt(f(x, y), 0.0),
        0.0,
        1.0
    )


def cone(x, y) -> fd.Function:
    x_0 = fd.Constant(0.5)
    y_0 = fd.Constant(0.5)
    r = fd.Constant(0.15)
    h = fd.Constant(1.0)

    return (
        h - fd.sqrt((pow(x - x_0, 2) + pow(y - y_0, 2))) / r
    )


def test_cone_integration() -> None:
    mesh = fd.UnitSquareMesh(10, 10)
    space_ls = fd.FunctionSpace(mesh, "CG", 1)
    ls = fd.Function(space_ls)
    x, y = fd.SpatialCoordinate(mesh)
    ls.interpolate(cone(x, y))

    fd.File("cone.pvd").write(ls)

    f = heaviside(cone, x, y)

    integral = integral_level_set(f, ls, dx)

    print(f"Exact: {fd.pi * 0.15 ** 2}")
    print(f"dx_integral: {assemble(f * dx)}")
    print(f"LS_integral: {assemble(integral)}")


def zalesak_disk(x, y) -> fd.Function:
    x_0 = fd.Constant(0.5)
    y_0 = fd.Constant(0.5)
    r = fd.Constant(0.15)
    h = fd.Constant(0.1)
    d = fd.Constant(0.05)

    return fd.conditional(
        fd.And(
            fd.lt(pow(x - x_0, 2) + pow(y - y_0, 2), pow(r, 2)),
            fd.Or(fd.gt(abs(x - x_0), d / 2), fd.gt(y, (y_0 + h)))
        ),
        1.0,
        0.0
    )


def zalesak_disk_ls(x, y) -> fd.Function:
    x_0 = fd.Constant(0.5)
    y_0 = fd.Constant(0.5)
    r = fd.Constant(0.15)
    h = fd.Constant(0.1)
    d = fd.Constant(0.05)

    return fd.conditional(
        fd.And(fd.lt(abs(x - x_0), d / 2), fd.lt(y, (y_0 + h))),
        -r,
        pow(r, 2) - (pow(x - x_0, 2) + pow(y - y_0, 2))
    )


def test_zalesak_integration() -> None:
    mesh = fd.UnitSquareMesh(10, 10)
    space_ls = fd.FunctionSpace(mesh, "CG", 1)
    ls = fd.Function(space_ls)
    x, y = fd.SpatialCoordinate(mesh)
    ls.interpolate(zalesak_disk_ls(x, y))

    fd.File("zalesak_ls.pvd").write(ls)

    f = heaviside(zalesak_disk_ls, x, y)

    integral = integral_level_set(f, ls, dx)

    print("Exact: 0.0582207")
    print(f"dx_integral: {assemble(f * dx)}")
    print(f"LS_integral: {assemble(integral)}")


def linear_function(x, y):
    return x + 5 * y - 4.11


def test_linear_function_integration() -> None:
    mesh = fd.UnitSquareMesh(16, 16)
    space_ls = fd.FunctionSpace(mesh, "CG", 1)
    ls = fd.Function(space_ls)
    x, y = fd.SpatialCoordinate(mesh)
    ls.interpolate(linear_function(x, y))

    f = heaviside(linear_function, x, y)

    integral = integral_level_set(f, ls, dx)

    print("Exact: 0.278")
    print(f"dx_integral: {assemble(f * dx(degree=1))}")
    print(f"LS_integral: {assemble(integral)}")


def test_helmholtz() -> None:
    mesh = fd.UnitSquareMesh(10, 10)

    x, y = fd.SpatialCoordinate(mesh)

    V = fd.FunctionSpace(mesh, "CG", 1)
    u = fd.TrialFunction(V)
    v = fd.TestFunction(V)
    ls = fd.Function(V)
    ls.interpolate(x + y - 1.31)

    f = fd.Function(V)
    f.interpolate(
        (1 + 8 * fd.pi * fd.pi) * fd.cos(x * fd.pi * 2) * fd.cos(y * fd.pi * 2))

    a_integrand = fd.inner(fd.grad(u), fd.grad(v)) + fd.inner(u, v)
    a = integral_level_set(a_integrand, ls, dx)
    L_integrand = fd.inner(f, v)
    L = integral_level_set(L_integrand, ls, dx)

    u = fd.Function(V)
    fd.solve(a == L, u, solver_parameters={'ksp_type': 'cg', 'pc_type': 'none'})

    fd.File("helmholtz.pvd").write(u)


if __name__ == "__main__":
    test_function_integration()
    test_linear_function_integration()
    test_cone_integration()
    test_zalesak_integration()
    test_helmholtz()
