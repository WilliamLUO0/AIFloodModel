module BG_cylc

include("GetTopNet.jl")
include("CollapseBG.jl")
include("gettide.jl")
include("makBGparam.jl")

export predictNZtides,MakBGParam,write2nc,CollapseBG,GetTopNetFlow,GetInjectionXY


end # module
