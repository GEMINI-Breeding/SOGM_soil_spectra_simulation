#!/usr/bin/env python
import numpy as np

from models.prospect import prospectall as prospect
from models.sail import FourSAIL as sail


def run_prosail(rsoil0,n, cab, car,  cbrown, cw, cm, lai, lidfa, hspot,
                tts, tto, psi, ant=0.0, alpha=40., prospect_version="5", 
                typelidf=2, lidfb=0., factor="SDR"):
    """Run the PROSPECT_5B and SAILh radiative transfer models. The soil
    model is a linear mixture model, where two spectra are combined together as

         rho_soil = rsoil*(psoil*soil_spectrum1+(1-psoil)*soil_spectrum2)
    By default, ``soil_spectrum1`` is a dry soil, and ``soil_spectrum2`` is a
    wet soil, so in that case, ``psoil`` is a surface soil moisture parameter.
    ``rsoil`` is a  soil brightness term. You can provide one or the two
    soil spectra if you want.  The soil spectra must be defined
    between 400 and 2500 nm with 1nm spacing.

    Parameters
    ----------
    rsoil0:
        The soil reflectance spectrum
    n: float
        Leaf layers
    cab: float
        leaf chlorophyll concentration
    car: float
        leaf carotenoid concentration
    cbrown: float
        senescent pigment
    cw: float
        equivalent leaf water
    cm: float
        leaf dry matter
    lai: float
        leaf area index
    lidfa: float
        a parameter for leaf angle distribution. If ``typliedf``=2, average
        leaf inclination angle.
    tts: float
        Solar zenith angle
    tto: float
        Sensor zenith angle
    psi: float
        Relative sensor-solar azimuth angle ( saa - vaa )
    ant: float
        leaf anthocyanin concentration (default set to 0)
    alpha: float
        The alpha angle (in degrees) used in the surface scattering
        calculations. By default it's set to 40 degrees.
    prospect_version: str
        Which PROSPECT version to use. We have "5" and "D"
    typelidf: int, optional
        The type of leaf angle distribution function to use. By default, is set
        to 2.
    lidfb: float, optional
        b parameter for leaf angle distribution. If ``typelidf``=2, ignored
    factor: str, optional
        What reflectance factor to return:
        * "SDR": directional reflectance factor (default)
        * "BHR": bi-hemispherical r. f.
        * "DHR": Directional-Hemispherical r. f. (directional illumination)
        * "HDR": Hemispherical-Directional r. f. (directional view)
        * "ALL": All of them


    Returns
    --------
    A reflectance factor between 400 and 2500 nm


    """

    factor = factor.upper()
    if factor not in ["SDR", "BHR", "DHR", "HDR", "ALL"]:
        raise ValueError("'factor' must be one of SDR, BHR, DHR, HDR or ALL")

    # expand rsoil0 to 2500 nm
    rsoil0 = np.append(rsoil0, rsoil0[-1])

    wv, refl, trans = prospect.run_prospect (n, cab, car,  cbrown, cw, cm, ant=ant,
                 prospect_version=prospect_version, alpha=alpha)
    
    [tss, too, tsstoo, rdd, tdd, rsd, tsd, rdo, tdo,
         rso, rsos, rsod, rddt, rsdt, rdot, rsodt, rsost, rsot,
         gammasdf, gammasdb, gammaso] = sail.foursailcore (refl, trans,
                                                  lidfa, lidfb, typelidf, 
                                                  lai, hspot, 
                                                  tts, tto, psi, rsoil0)



    if factor == "SDR":
        return rsot
    elif factor == "BHR":
        return rddt
    elif factor == "DHR":
        return rsdt
    elif factor == "HDR":
        return rdot
    elif factor == "ALL":
        return [rsot, rddt, rsdt, rdot]

        
