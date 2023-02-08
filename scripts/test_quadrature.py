import firedrake as fd
import numpy as np
from finat.point_set import PointSet
from finat.quadrature import QuadratureRule, make_quadrature
from firedrake import READ, WRITE, assemble, dx, par_loop
from ufl.core.expr import Expr


def integral_level_set(f: Expr, ls: fd.Function, dx: fd.Measure, mesh: fd.Mesh) -> fd.Form:
    """ This function modifies the quadrature points in the elements sliced by
    the level-set function. It returns a Form as a sum of integrals.

    Args:
        f (Expr): Integrand.
        ls (Function): The level-set function. It should be CG(1) function.
        dx (Measure):  Integral measure.
        mesh (Mesh): Domain mesh.

    Returns:
        Form: A firedrake form will be returned, which can then be assembled.

    """
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

            double node_flag[3] = { 1,1,1 };	//
            double element_flag = 0;			//

            // Quadrature rule
            int n_q = 3;
            double w_q[3] = { 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0 };
            double x_q[3] = { 0, 0.5, 0.5 };
            double y_q[3] = { 0.5, 0, 0.5 };


            // Flag every nodes and gives a binary representation of the level set function
            // 0,7 -> nothing happens (all negative reps. all positive)
            // 1,2,4 -> First, second resp. third node are positive
            // 3,5,6 -> Third, second resp. first node are negative

            for (int i = 0; i < 3; i++) {

                min_value = fmin(0, a[i]);

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

                crossing_point_x[0] = -a[0] / (a[1] - a[0]); // a[1] and a[0] are different because of the different sign
                crossing_point_y[1] = -a[0] / (a[2] - a[0]); // a[2] and a[0] are different because of the different sign

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

                crossing_point_x[0] = -a[0] / (a[1] - a[0]);	// See above
                crossing_point_x[1] = -a[2] / (a[1]-a[2]);
                crossing_point_y[1] = a[1] / (a[1] - a[2]);

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

                crossing_point_x[0] = -a[2] / (a[1] - a[2]);	// See above
                crossing_point_y[0] = a[1] / (a[1] - a[2]);
                crossing_point_y[1] = -a[0] / (a[2] - a[0]);

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
                    w_q_triangle[j][k] = det_triangle[j] * w_q[k];
                    x_q_triangle[j][k] = B_triangle[j][0][0] * x_q[k] + B_triangle[j][0][1] * y_q[k] + b_triangle[j][0];
                    y_q_triangle[j][k] = B_triangle[j][1][0] * x_q[k] + B_triangle[j][1][1] * y_q[k] + b_triangle[j][1];
                }

            }

            for (int i=0; i<a.dofs; i++) {
                min_value = fmin(min_value, a[i]);
                max_value = fmax(max_value, a[i]);
            }
            if (min_value < 0 && max_value > 0) {
                b[0] = 1.0;
            }
            else {
                b[0] = 0.0;
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
            'a': (ls, READ),
            'b': (quad_indicator, WRITE),
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

            form += f * marker * dx(rule=rule2D)

    return form


def test_function_integration() -> None:
    mesh = fd.UnitSquareMesh(2, 2)
    S = fd.FunctionSpace(mesh, "CG", 1)
    f = fd.Function(S)
    ls = fd.Function(S)

    x, y = fd.SpatialCoordinate(mesh)
    f.interpolate(x)
    ls.interpolate(x + y - 1.31)

    integral = integral_level_set(f, ls, dx, mesh)

    print(assemble(integral))

    print(assemble(f * dx))


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
    a = integral_level_set(a_integrand, ls, dx, mesh)
    L_integrand = fd.inner(f, v)
    L = integral_level_set(L_integrand, ls, dx, mesh)

    u = fd.Function(V)
    fd.solve(a == L, u, solver_parameters={'ksp_type': 'cg', 'pc_type': 'none'})

    fd.File("helmholtz.pvd").write(u)


if __name__ == "__main__":
    test_function_integration()
    test_helmholtz()
