#include "easi/component/AndersonianStress.h"  // Includes the header file for the AndersonianStress class.

#include <cmath>  // Includes the cmath library for mathematical functions like sin, cos, and atan.

namespace easi {  // Defines the namespace easi, which encapsulates the AndersonianStress class.

void AndersonianStress::evaluate() {  // Defines the evaluate method for the AndersonianStress class.
    double const pi = 3.141592653589793;  // Defines the constant value of pi.
    
    // Calculates the most favorable direction for stress.
    double Phi = pi / 4.0 - 0.5 * atan(i.mu_s);  // Phi is the angle most favorable for slip computed using the friction coefficient (mu_s).
    double SHmax_rad = i.SH_max * pi / 180.0;  // Converts the maximum horizontal stress direction from degrees to radians.
    double s2 = sin(2.0 * Phi);  // Calculates the sine of 2*Phi., 
    double c2 = cos(2.0 * Phi);  // Calculates the cosine of 2*Phi.
    double alpha = (2.0 * i.s2ratio - 1.0) / 3.0;  // Computes alpha based on the stress ratio.
    
    // Assumes effective confining stress as the absolute value of the vertical stress component.
    double effectiveConfiningStress = std::fabs(i.sig_zz);
    
    // Computes R using the stress shape ratio (S).
    double R = 1.0 / (1.0 + i.S);
    
    // Computes the differential stress (ds).
    double ds = (i.mu_d * effectiveConfiningStress +
                 R * (i.cohesion + (i.mu_s - i.mu_d) * effectiveConfiningStress)) /
                (s2 + i.mu_d * (alpha + c2) + R * (i.mu_s - i.mu_d) * (alpha + c2));
    
    // Calculates the mean stress.
    double sm = effectiveConfiningStress - alpha * ds;
    
    // Computes the stress components sii, ensuring they are positive.
    double s11 = sm + ds;
    double s22 = sm - ds + 2.0 * ds * i.s2ratio;
    double s33 = sm - ds;
    
    // Determines the principal stress components based on S_v.
    double S_hmax, S_hmin, S_v;
    if (i.S_v == 1) {
        S_hmax = s22;
        S_hmin = s33;
        S_v = s11;
    } else {
        if (i.S_v == 2) {
            S_hmax = s11;
            S_hmin = s33;
            S_v = s22;
        } else {
            S_hmax = s11;
            S_hmin = s22;
            S_v = s33;
        }
    }

    // Computes cosine and sine of SHmax_rad.
    double cs = cos(SHmax_rad);
    double ss = sin(SHmax_rad);

    // Calculates normalized stress components using SeisSol's convention (compressive stress < 0).
    o.b_xx = -(cs * cs * S_hmin + ss * ss * S_hmax) * effectiveConfiningStress / S_v;
    o.b_xy = -cs * ss * (S_hmax - S_hmin) * effectiveConfiningStress / S_v;
    o.b_xz = 0;
    o.b_yy = -(ss * ss * S_hmin + cs * cs * S_hmax) * effectiveConfiningStress / S_v;
    o.b_yz = 0;
    o.b_zz = -effectiveConfiningStress;  // Vertical stress component.

}

}  // End of namespace easi.

