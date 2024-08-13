#include "easi/component/AndersonianStress.h"

#include <cmath> // 

namespace easi {

// COMPUTE NORMALIZED STRESS FOLLOWING THE METHOD OF Ulrich et al. (2018)
void AndersonianStress::evaluate() {
    double const pi = 3.141592653589793;
    // most favorable direction (A4, AM2003)
    double Phi = pi / 4.0 - 0.5 * atan(i.mu_s);
    double SHmax_rad = i.SH_max * pi / 180.0;
    double s2 = sin(2.0 * Phi);
    double c2 = cos(2.0 * Phi);
    double alpha = (2.0 * i.s2ratio - 1.0) / 3.0;
    // we assume that szz is the effectiveConfiningStress  = sii/3 which is not necessarily true,
    // but then we scale the stress to get b_zz = szz
    double effectiveConfiningStress = std::fabs(i.sig_zz);
    double R = 1.0 / (1.0 + i.S);
    double ds = (i.mu_d * effectiveConfiningStress +
                 R * (i.cohesion + (i.mu_s - i.mu_d) * effectiveConfiningStress)) /
                (s2 + i.mu_d * (alpha + c2) + R * (i.mu_s - i.mu_d) * (alpha + c2));
    double sm = effectiveConfiningStress - alpha * ds;
    // sii are all positive
    double s11 = sm + ds;
    double s22 = sm - ds + 2.0 * ds * i.s2ratio;
    double s33 = sm - ds;
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

    double cs = cos(SHmax_rad);
    double ss = sin(SHmax_rad);

    // minus signs added to fall back to SeisSol convention (compressive stress<0)

    o.b_xx = -(cs * cs * S_hmin + ss * ss * S_hmax) * effectiveConfiningStress / S_v;
    o.b_xy = -cs * ss * (S_hmax - S_hmin) * effectiveConfiningStress / S_v;
    o.b_xz = 0;
    o.b_yy = -(ss * ss * S_hmin + cs * cs * S_hmax) * effectiveConfiningStress / S_v;
    o.b_yz = 0;
    o.b_zz = -effectiveConfiningStress;
}

} // namespace easi
