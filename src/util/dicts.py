import numpy.typing as npt
from typing import Callable, NotRequired, Union, Tuple, Any, Optional, TypedDict

class AnalyInputOpt(TypedDict):
    model_type : NotRequired[str]   
    mater_calib : NotRequired[str]  
    bar_cm : NotRequired[Callable[[npt.NDArray, bool], Tuple[npt.NDArray, npt.NDArray, Optional[npt.NDArray]]]]
    rot_spr_bend : NotRequired[Callable[[npt.NDArray, npt.NDArray, Any, npt.NDArray, bool], Tuple[npt.NDArray, npt.NDArray, Optional[npt.NDArray]]]]
    rot_spr_fold : NotRequired[Callable[[npt.NDArray, npt.NDArray, Any, npt.NDArray, bool], Tuple[npt.NDArray, npt.NDArray, Optional[npt.NDArray]]]]
    a_bar : NotRequired[Union[npt.NDArray, float]]
    K_b : NotRequired[Union[npt.NDArray, float]]
    K_f : NotRequired[Union[npt.NDArray, float]]
    mod_elastic : NotRequired[float]
    poisson : NotRequired[float]
    thickness : NotRequired[float]
    l_scale_factor : NotRequired[float]
    zero_bend : NotRequired[Union[str, float, npt.NDArray]]
    load_type : NotRequired[str]
    load : NotRequired[npt.NDArray]
    adaptive_load : NotRequired[Callable[[npt.NDArray, npt.NDArray, int], npt.NDArray]]
    initial_load_factor : NotRequired[float]
    max_incr : NotRequired[int]
    disp_step : NotRequired[int]
    stop_criterion : NotRequired[Callable[[npt.NDArray, npt.NDArray, float], bool]]

class Truss(TypedDict):
    node : npt.NDArray
    bars : npt.NDArray
    trigl : npt.NDArray
    b : npt.NDArray
    l : npt.NDArray
    fixed_dofs : npt.NDArray
    cm : Callable[[Any, Any], Any]
    a : npt.NDArray
    u_0 : Optional[npt.NDArray]

class Angles(TypedDict):
    panel : npt.NDArray
    fold : npt.NDArray
    bend : npt.NDArray
    pf_0 : npt.NDArray
    pb_0 : npt.NDArray
    cm_bend : Callable[[npt.NDArray, npt.NDArray, Any, npt.NDArray, bool], Tuple[npt.NDArray, npt.NDArray, Optional[npt.NDArray]]]
    cm_fold : Callable[[npt.NDArray, npt.NDArray, Any, npt.NDArray, bool], Tuple[npt.NDArray, npt.NDArray, Optional[npt.NDArray]]]
    k_b : npt.NDArray | None
    k_f : npt.NDArray  



class Bar(TypedDict):
    ex : npt.NDArray
    sx : npt.NDArray
    us_i : npt.NDArray
    us : npt.NDArray

class Fold(TypedDict):
    angle : npt.NDArray
    rm : npt.NDArray
    uf_i : npt.NDArray
    uf : npt.NDArray

class Bend(TypedDict):
    angle : npt.NDArray
    rm : npt.NDArray
    ub_i : npt.NDArray
    ub : npt.NDArray

class Stat(TypedDict):
    bar: Bar
    fold : Fold
    bend : Bend
    pe : npt.NDArray