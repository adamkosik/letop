import firedrake as fd
from firedrake import dx, assemble
from firedrake import par_loop, READ, WRITE
from finat.point_set import PointSet
import finat
import numpy as np

mesh = fd.UnitSquareMesh(2, 2)
S = fd.FunctionSpace(mesh, "CG", 1)
ls = fd.Function(S)

x, y = fd.SpatialCoordinate(mesh)
ls.interpolate(x + y - 1.3)


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

        double a_shift[3];

        for (int i = 0; i < 3; i++) {
            if (i != 2) {
                a_shift[i] = a[i + 1];
            }
            else {
                a_shift[i] = a[0];
            }

        }

        double node_flag[3] = { 1,1,1 };	//  
        double element_flag = 0;			// 

        // Quadrature rule
        double w_q[3] = { 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0 };
        double x_q[3] = { 0, 0.5, 0.5 };
        double y_q[3] = { 0.5, 0, 0.5 };

        // Flag every nodes and gives a binary representation of the level set function
        // 0,7 -> nothing happens (all negative reps. all positive)
        // 1,2,4 -> First, second resp. third node are positive
        // 3,5,6 -> Third, second resp. first node are negative
        
        for (int i = 0; i < 3; i++) {

            min_value = fmin(0, a_shift[i]);

            if (min_value < 0) {
                node_flag[i] = 0;
            }

            element_flag += node_flag[i]* pow(2, i);

        }

        // To compute where the level set crosses the reference element the following cases are equivalent
        // 1 and 6, 2 and 5, 3 and 4
        double crossing_point_x[2] = { 0.0 };
        double crossing_point_y[2] = { 0.0 };

        double triangle_1_x[3] = { 0.0 };
        double triangle_1_y[3] = { 0.0 };
        double det_triangle_1 = 0.0;
        double B_triangle_1[2][2] = { 0.0 };
        double b_triangle_1[2] = { 0.0 };
        double w_q_triangle_1[3] = { 0.0 };			// Tranformed quadrature rule for the first triangle
        double x_q_triangle_1[3] = { 0.0};
        double y_q_triangle_1[3] = { 0.0 };

        double triangle_2_x[3] = { 0.0 };
        double triangle_2_y[3] = { 0.0 };
        double det_triangle_2 = 0.0;
        double B_triangle_2[2][2] = { 0.0 };
        double b_triangle_2[2] = { 0.0 };
        double w_q_triangle_2[3] = { 0.0 };			// Tranformed quadrature rule for the second triangle
        double x_q_triangle_2[3] = { 0.0 };
        double y_q_triangle_2[3] = { 0.0 };

        double triangle_3_x[3] = { 0.0 };
        double triangle_3_y[3] = { 0.0 };
        double det_triangle_3 = 0.0;
        double B_triangle_3[2][2] = { 0.0 };
        double b_triangle_3[2] = { 0.0 };
        double w_q_triangle_3[3] = { 0.0 };			// Tranformed quadrature rule for the third triangle
        double x_q_triangle_3[3] = { 0.0 };
        double y_q_triangle_3[3] = { 0.0 };

        if (element_flag == 1 || element_flag == 6) {

            crossing_point_x[0] = -a_shift[0] / (a_shift[1] - a_shift[0]); // a_shift[1] and a_shift[0] are different because of the different sign 
            crossing_point_y[1] = -a_shift[0] / (a_shift[2] - a_shift[0]); // a_shift[2] and a_shift[0] are different because of the different sign

            // Transformation for the first triangle
            triangle_1_x[1] = crossing_point_x[0];
            triangle_1_y[2] = crossing_point_y[1];
            det_triangle_1 = crossing_point_x[0] * crossing_point_y[1];


            // Transformation for the second triangle
            triangle_2_x[0] = crossing_point_x[0];
            triangle_2_y[1] = crossing_point_y[1];
            triangle_2_x[2] = 1.0;
            det_triangle_2 = (1-crossing_point_y[1])*crossing_point_x[0];


            // Transformation for the third triangle
            triangle_3_x[1] = 1.0;
            triangle_3_y[0] = crossing_point_y[1];
            triangle_3_y[2] = 1.0;
            det_triangle_3 = 1 - crossing_point_y[1];
            
            

        }

        if (element_flag == 2 || element_flag == 5) {

            crossing_point_x[0] = -a_shift[0] / (a_shift[1] - a_shift[0]);	// See above
            crossing_point_x[1] = -a_shift[2] / (a_shift[1]-a_shift[2]);
            crossing_point_y[1] = a_shift[1] / (a_shift[1] - a_shift[2]);

            triangle_1_x[0] = crossing_point_x[0];
            triangle_1_x[1] = 1.0;
            triangle_1_x[2] = crossing_point_x[1];
            triangle_1_y[2] = crossing_point_y[1];
            det_triangle_1 = (1-crossing_point_x[0]) * crossing_point_y[1];

            triangle_2_x[1] = crossing_point_x[0];
            triangle_2_x[2] = crossing_point_x[1];
            triangle_2_y[2] = crossing_point_y[1];
            det_triangle_2 = crossing_point_x[0]* crossing_point_y[1];

            triangle_3_x[1] = crossing_point_x[1];
            triangle_3_y[1] = crossing_point_y[1];
            triangle_3_y[2] = 1.0;
            det_triangle_3 = 1 - crossing_point_y[1];

        }

        if (element_flag == 3 || element_flag == 4) {

            crossing_point_x[0] = -a_shift[2] / (a_shift[1] - a_shift[2]);	// See above
            crossing_point_y[0] = a_shift[1] / (a_shift[1] - a_shift[2]);
            crossing_point_y[1] = -a_shift[0] / (a_shift[2] - a_shift[0]);

            triangle_1_y[0] = crossing_point_y[1];
            triangle_1_x[1] = crossing_point_x[0];
            triangle_1_y[1] = crossing_point_y[0];
            triangle_1_y[2] = 1;
            det_triangle_1 = (1 - crossing_point_x[0]) * crossing_point_y[1];

            triangle_2_x[1] = crossing_point_x[0];
            triangle_2_y[1] = crossing_point_y[0];
            triangle_2_y[2] = crossing_point_y[1];
            det_triangle_2 = crossing_point_x[0] * crossing_point_y[1];

            triangle_3_x[1] = 1;
            triangle_3_x[2] = crossing_point_x[0];
            triangle_3_y[2] = crossing_point_y[0];
            det_triangle_3 = 1 - crossing_point_y[0];

        }

        // Matrix and vector for transformation to the first triangle
        b_triangle_1[0] = triangle_1_x[0];
        b_triangle_1[1] = triangle_1_y[0];
        B_triangle_1[0][0] = triangle_1_x[1] - triangle_1_x[0];
        B_triangle_1[1][0] = triangle_1_y[1] - triangle_1_y[0];
        B_triangle_1[0][1] = triangle_1_x[2] - triangle_1_x[0];
        B_triangle_1[1][1] = triangle_1_y[2] - triangle_1_y[0];
        
        // Matrix and vector for transformation to the second triangle
        b_triangle_2[0] = triangle_2_x[0];
        b_triangle_2[1] = triangle_2_y[0];
        B_triangle_2[0][0] = triangle_2_x[1] - triangle_2_x[0];
        B_triangle_2[1][0] = triangle_2_y[1] - triangle_2_y[0];
        B_triangle_2[0][1] = triangle_2_x[2] - triangle_2_x[0];
        B_triangle_2[1][1] = triangle_2_y[2] - triangle_2_y[0];

        // Matrix and vector for transformation to the third triangle
        b_triangle_3[0] = triangle_3_x[0];
        b_triangle_3[1] = triangle_3_y[0];
        B_triangle_3[0][0] = triangle_3_x[1] - triangle_3_x[0];
        B_triangle_3[1][0] = triangle_3_y[1] - triangle_3_y[0];
        B_triangle_3[0][1] = triangle_3_x[2] - triangle_3_x[0];
        B_triangle_3[1][1] = triangle_3_y[2] - triangle_3_y[0];

        // Tranformation of the quadrature rule to the first sub triangle
        for (int j = 0; j < 3; j++) { 
            w_q_triangle_1[j] = det_triangle_1 * w_q[j];  
            x_q_triangle_1[j] = B_triangle_1[0][0] * x_q[j] + B_triangle_1[0][1] * y_q[j] + b_triangle_1[0];
            y_q_triangle_1[j] = B_triangle_1[1][0] * x_q[j] + B_triangle_1[1][1] * y_q[j] + b_triangle_1[1];
        }

        // Tranformation of the quadrature rule to the second sub triangle
        for (int j = 0; j < 3; j++) {
            w_q_triangle_2[j] = det_triangle_2 * w_q[j];  
            x_q_triangle_2[j] = B_triangle_2[0][0] * x_q[j] + B_triangle_2[0][1] * y_q[j] + b_triangle_2[0];
            y_q_triangle_2[j] = B_triangle_2[1][0] * x_q[j] + B_triangle_2[1][1] * y_q[j] + b_triangle_2[1];
        }

        // Tranformation of the quadrature rule to the third sub triangle
        for (int j = 0; j < 3; j++) {
            w_q_triangle_3[j] = det_triangle_3 * w_q[j];  
            x_q_triangle_3[j] = B_triangle_3[0][0] * x_q[j] + B_triangle_3[0][1] * y_q[j] + b_triangle_3[0];
            y_q_triangle_3[j] = B_triangle_3[1][0] * x_q[j] + B_triangle_3[1][1] * y_q[j] + b_triangle_3[1];
        }

        int test = 0;

        for (int i=0; i<a.dofs; i++) {
            min_value = fmin(min_value, a_shift[i]);
            max_value = fmax(max_value, a_shift[i]);
        }
        if (min_value < 0 && max_value > 0) {
            b[0] = 1.0;
        }
        else {
            b[0] = 0.0;
        }
        for (int j = 0; j < 3; j++) {
            w[j] = w_q_triangle_1[j];
            x[j] = x_q_triangle_1[j];
            y[j] = y_q_triangle_1[j];
            w[j+3] = w_q_triangle_2[j];
            x[j+3] = x_q_triangle_2[j];
            y[j+3] = y_q_triangle_2[j];
            w[j+6] = w_q_triangle_3[j];
            x[j+6] = x_q_triangle_3[j];
            y[j+6] = y_q_triangle_3[j];
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

integral = 0

print(quad_indicator.dat.data)

inv_indicator = fd.Function(space_indicator)
inv_indicator.assign(1.0)

for idx, quadrature in enumerate(quad_indicator.dat.data):
    if quad_indicator.dat.data[idx] == 1.0:
        marker = fd.Function(space_indicator)
        marker.assign(0.0)
        marker.dat.data[idx] = 1.0
        inv_indicator.dat.data[idx] = 0.0

        print(quad_y.dat.data[idx])

        points1 = PointSet(
            np.stack((quad_x.dat.data[idx], quad_y.dat.data[idx]), axis=-1)
        )
        weights1 = quad_weights.dat.data[idx]
        rule2D = finat.quadrature.QuadratureRule(points1, weights1)
        integral += assemble(x * marker * dx(rule=rule2D))
    print(integral)


integral += assemble(x * inv_indicator * dx)

print(integral)
print(assemble(x * dx))
fd.File("ls.pvd").write(ls)

integral += assemble(x * inv_indicator * dx)
