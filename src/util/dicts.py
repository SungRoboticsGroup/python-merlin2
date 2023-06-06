import numpy.typing as npt
from typing import Callable, NotRequired, Union, Tuple, Any, Optional, TypedDict

class AnalyInputOpt(TypedDict):
    model_type : NotRequired[str]   
    mater_calib : NotRequired[str]  
    bar_cm : NotRequired[Callable[[npt.NDArray, bool], Tuple[npt.NDArray, npt.NDArray, Optional[npt.NDArray]]]]
    rot_spr_bend : NotRequired[Callable[[npt.ArrayLike, npt.ArrayLike, Any, npt.ArrayLike], Tuple[npt.ArrayLike, npt.ArrayLike, npt.ArrayLike]]]
    rot_spr_fold : NotRequired[Callable[[npt.ArrayLike, npt.ArrayLike, Any, npt.ArrayLike], Tuple[npt.ArrayLike, npt.ArrayLike, npt.ArrayLike]]]
    a_bar : NotRequired[Union[npt.ArrayLike, float]]
    K_b : NotRequired[npt.ArrayLike]
    K_f : NotRequired[npt.ArrayLike]
    mod_elastic : NotRequired[float]
    poisson : NotRequired[float]
    thickness : NotRequired[float]
    l_scale_factor : NotRequired[float]
    zero_bend : NotRequired[Union[str, float, npt.ArrayLike]]
    load_type : NotRequired[str]
    load : NotRequired[npt.NDArray]
    adaptive_load : NotRequired[Callable[[npt.ArrayLike, npt.ArrayLike, float], npt.ArrayLike]]
    initial_load_factor : NotRequired[float]
    max_incr : NotRequired[int]
    disp_step : NotRequired[int]
    stop_criterion : NotRequired[Callable[[npt.ArrayLike, npt.ArrayLike, float], bool]]

class Truss(TypedDict):
    node : npt.ArrayLike
    bars : npt.ArrayLike
    trigl : npt.ArrayLike
    b : npt.ArrayLike
    l : npt.ArrayLike
    fixed_dofs : npt.ArrayLike
    cm : Callable[[Any, Any, Any], Any]
    a : npt.ArrayLike
    u_0 : npt.ArrayLike

class Angles(TypedDict):
    panel : npt.ArrayLike
    fold : npt.ArrayLike
    bend : npt.ArrayLike
    pf_0 : npt.ArrayLike
    pb_0 : npt.ArrayLike
    cm_bend : Callable[[npt.ArrayLike, npt.ArrayLike, Any, npt.ArrayLike], Tuple[npt.ArrayLike, npt.ArrayLike, npt.ArrayLike]]
    cm_fold : Callable[[npt.ArrayLike, npt.ArrayLike, Any, npt.ArrayLike], Tuple[npt.ArrayLike, npt.ArrayLike, npt.ArrayLike]]
    k_b : npt.ArrayLike
    k_f : npt.ArrayLike  