using DelimitedFiles, Printf, NetCDF,Dates,CFTime

mutable struct River
	xmin::Float64
	xmax::Float64
	ymin::Float64
	ymax::Float64
	flowfile::String
end
mutable struct Bnd
	ttype::Int64
	file::String
end

mutable struct Bndseg
	polyfile::String
	ttype::Int64
	file::String
end

mutable struct modelinfo
	region::NTuple{4, Float64}
	dx::Float64
	demfile::String
	roughnessfile::String
	polyfile::String
	rainfile::String
	injectionfile::String
	injectionfolder::String
	levels::NTuple{3, Int64}
	targetlevelfile::String
	tidefile::String
	BGoutfile::String
	tidepoly::String
	coordinate::NTuple{2, Float64}
	timepicflood::String
	reftime::String
	starttime::String
	zsinit::Float64
	slr::Float64
	


end

mutable struct Params

	bathy::String
	gpudevice::Int64

	region::NTuple{4, Float64}

	dx::Float64

	domain::String

	rivers::Vector{River}

	rain::String

	cfmap::String

	#Output
	outstep::Float64
	outfile::String
	outvars::Vector{String}
	smallnc::Bool

	#adapt
	levels::NTuple{3, Float64}
	targetlevelfile::String

	#bnd
	bnds::Vector{Bndseg}

	#timing
	reftime::String
	starttime::String
	endtime::String

	#Initial condition and SLR
	zsinit::Float64
	slr::Float64

end
	function WriteBGparam(outparamfile::String, BGParam::Params)
		# Something

		open(outparamfile,"w") do io

			Printf.@printf(io,"##########\n");
			Printf.@printf(io,"# DEM    #\n");
			Printf.@printf(io,"##########\n\n");

			Printf.@printf(io,"dem = %s\n\n",BGParam.bathy)

			if !isnan(BGParam.region[1])
				Printf.@printf(io,"xmin = %f\n",BGParam.region[1])
			end
			if !isnan(BGParam.region[2])
				Printf.@printf(io,"xmax = %f\n",BGParam.region[2])
			end
			if !isnan(BGParam.region[3])
				Printf.@printf(io,"ymin = %f\n",BGParam.region[3])
			end
			if !isnan(BGParam.region[4])
				Printf.@printf(io,"ymax = %f\n",BGParam.region[4])
			end

			if !isempty(BGParam.domain)
				Printf.@printf(io,"aoi = %s\n",BGParam.domain)
			end


            if !isnan(BGParam.dx)
                Printf.@printf(io,"dx = %f\n\n",BGParam.dx)
			end

			Printf.@printf(io,"##############\n");
			Printf.@printf(io,"# Forcing    #\n");
			Printf.@printf(io,"##############\n\n");

			if !isnan(BGParam.slr)
				Printf.@printf(io,"zsoffset = %f\n",BGParam.slr)
				#zsoffset also offsets zsinit. in this particular case it is undesirable so we make sure zsinit is adjusted accordingly
				BGParam.zsinit = BGParam.zsinit - BGParam.slr
			end

			if !isnan(BGParam.zsinit)
				Printf.@printf(io,"zsinit = %f\n",BGParam.zsinit)
			end
			

			if !isempty(BGParam.cfmap)
				Printf.@printf(io,"frictionmodel = %d\n",1)
				Printf.@printf(io,"cfmap = %s\n\n",BGParam.cfmap)
			end

			if !isempty(BGParam.rain)
				Printf.@printf(io,"rainfile =%s\n\n",BGParam.rain)
			end

			Printf.@printf(io,"##########\n");
			Printf.@printf(io,"# Rivers #\n");
			Printf.@printf(io,"##########\n\n");
			for ir=1:length(BGParam.rivers)
				Printf.@printf(io,"river = %s,%f,%f,%f,%f\n",BGParam.rivers[ir].flowfile,BGParam.rivers[ir].xmin,BGParam.rivers[ir].xmax,BGParam.rivers[ir].ymin,BGParam.rivers[ir].ymax)
			end

			Printf.@printf(io,"\n\n");



			#Printf.@printf(io,"test =%d\n",BGParam.test)

			Printf.@printf(io,"###############\n");
			Printf.@printf(io,"# Adaptation  #\n");
			Printf.@printf(io,"###############\n\n");
			Printf.@printf(io,"initlevel = %d\n",BGParam.levels[3])

			#Printf.@printf(io,"outishift =%d\n",BGParam.outishift)
			#Printf.@printf(io,"outjshift =%d\n",BGParam.outjshift)

			if BGParam.levels[1] != BGParam.levels[2]

				Printf.@printf(io,"Adaptation = Targetlevel,%s\n",BGParam.targetlevelfile)
				Printf.@printf(io,"minlevel = %d\n",BGParam.levels[1])
				Printf.@printf(io,"maxlevel = %d\n\n",BGParam.levels[2])
			end

			Printf.@printf(io,"################\n");
			Printf.@printf(io,"# Boundaries   #\n");
			Printf.@printf(io,"################\n\n");

			Printf.@printf(io,"aoibnd = 3;\n");

			for i=1:length(BGParam.bnds)
				Printf.@printf(io,"bndseg = %s,%s,%d;\n",BGParam.bnds[i].polyfile,BGParam.bnds[i].file,BGParam.bnds[i].ttype);
			end



			Printf.@printf(io,"#############\n");
			Printf.@printf(io,"# Timing    #\n");
			Printf.@printf(io,"#############\n\n");

			Printf.@printf(io,"bndtaper = %f\n\n",2*3600.0)
			if !isempty(BGParam.reftime)
				Printf.@printf(io,"reftime = %s\n\n",BGParam.reftime)
			end

            if !isempty(BGParam.starttime)
				Printf.@printf(io,"starttime = %s\n\n",BGParam.starttime)
			end

            if BGParam.outstep>0.0
				Printf.@printf(io,"outputtimestep = %f\n\n",BGParam.outstep)
            end			

            if !isempty(BGParam.endtime)
				Printf.@printf(io,"endtime = %s\n\n",BGParam.endtime)
			end

			Printf.@printf(io,"#############\n");
			Printf.@printf(io,"# Others    #\n");
			Printf.@printf(io,"#############\n\n");
			Printf.@printf(io,"gpudevice = %d\n\n",BGParam.gpudevice)
			Printf.@printf(io,"vmax = %f\n\n",10.0)


			Printf.@printf(io,"#############\n");
			Printf.@printf(io,"# Output    #\n");
			Printf.@printf(io,"#############\n\n");

			Printf.@printf(io,"smallnc = %d\n\n",0)
			Printf.@printf(io,"savebyblk= %s\n\n","false")

			if !isempty(BGParam.outvars)
				Printf.@printf(io,"outvars = ")
				for ir=1:length(BGParam.outvars)
					Printf.@printf(io,"%s",BGParam.outvars[ir]);
					if(ir==length(BGParam.outvars))
						Printf.@printf(io,";\n\n")
					else
						Printf.@printf(io,", ")
					end

				end
			end

			if !isempty(BGParam.outfile)
				Printf.@printf(io,"outfile = %s\n\n",BGParam.outfile)
			end



		end




	end

	function GetParams(;dem="",vars=Vector{String}(undef, 0), bndsegs=Vector{Bndseg}(undef, 0),rivers=Vector{River}(undef, 0), gpu=0,rainfile="",roughness="",region=(NaN64,NaN64,NaN64,NaN64),dx=NaN64, levels=(0,0,0),targetlevelfile="",polygonfile="",reftime="",starttime="",endtime="",zsinit=NaN64,slr=NaN64)

		return Params(dem,gpu,region,dx,polygonfile,rivers,rainfile,roughness,0,"",vars,false,levels,targetlevelfile,bndsegs,reftime,starttime,endtime,zsinit,slr)
	end
	function readinfo(infofile::String)

		myinfo=modelinfo((NaN64,NaN64,NaN64,NaN64),NaN64,"","","","","","",(0,0,0),"","","","",(NaN64,NaN64),"","","",NaN64,NaN64);


		open(infofile) do f
			regstr=readline(f)
			dxstr=readline(f)

	    	myinfo.demfile=readline(f)
			myinfo.roughnessfile=readline(f)
			myinfo.polyfile=readline(f)
			myinfo.rainfile=readline(f)
			myinfo.injectionfile=readline(f)
			myinfo.injectionfolder=readline(f)
			levstr=readline(f)
			myinfo.targetlevelfile=readline(f)
			myinfo.tidefile=readline(f)
			myinfo.BGoutfile=readline(f)
			
			coordstr=readline(f)
			#Tpic=readline(f)# That's not Tpic?
			myinfo.timepicflood=readline(f)
			myinfo.reftime=readline(f)
			myinfo.starttime=readline(f)

			#myinfo.tidepoly=readline(f)
			tidepoly=readline(f)

			myinfo.zsinit=parse(Float64,readline(f))
			myinfo.slr=parse(Float64,readline(f))

			reg=split(regstr,",")

			myinfo.region=(parse(Float64,reg[1]),parse(Float64,reg[2]),parse(Float64,reg[3]),parse(Float64,reg[4]))
			myinfo.dx=parse(Float64,dxstr)

			levels=split(levstr,",")
			myinfo.levels=(parse(Int64,levels[1]),parse(Int64,levels[2]),parse(Int64,levels[3]))

            coords=split(coordstr,"\t")
			myinfo.coordinate=(parse(Float64,coords[1]),parse(Float64,coords[2]))

			#myinfo.timepicflood=parse(Float64,Tpic)
			
			if !isempty(tidepoly)
				myinfo.tidepoly=tidepoly
			end

			




		end

		return myinfo

	end


function readinjectionpoints(infile::String;outfolder="")
	
	
	rivers=Vector{River}(undef, 0);

	data = Array{Any}(undef, 0, 0);

	try
		data = readdlm(infile,',')
	catch
		data = Array{Any}(undef, 0, 0)
	end


	if length(data)>0

		id=Int64.(data[:,1]);
		x=data[:,2];
		y=data[:,3];
		Name=String.(data[:,4]);

		

		for ir=1:length(id)

			idst=Printf.@sprintf("%d",id[ir])


			push!(rivers,River(x[ir]-5.0,x[ir]+5.0,y[ir]-5.0,y[ir]+5.0,outfolder*Name[ir]*".txt"))
		end
	end
	return rivers
end

function MakBGParam(infofile,outfile; Outfolder="", coarse=false)
	myinfo=readinfo(infofile);
	myrivers=readinjectionpoints(myinfo.injectionfile,outfolder=myinfo.injectionfolder);
	myparam=GetParams(dem=myinfo.demfile,roughness=myinfo.roughnessfile*"?zo",rivers=myrivers,rainfile=myinfo.rainfile,vars=["zs", "u", "v", "h", "hmax", "zsmax", "U", "hUmax", "Umax"],polygonfile=myinfo.polyfile,reftime=myinfo.reftime,starttime=myinfo.starttime,zsinit=myinfo.zsinit,slr=myinfo.slr);

	myparam.region=myinfo.region;
	myparam.dx=myinfo.dx;

	

	if coarse
		# Make all level the same
		myparam.levels=(myinfo.levels[1],myinfo.levels[1],myinfo.levels[1])

		# Append the _coarse before the .nc
		myparam.outfile = myinfo.BGoutfile * "_coarse.nc"
               
		# Remove the rain for the coarse run
		#myparam.rain = ""

		# Change the uniform resoltion to 32m
		myparam.dx = 32

		# Remove intermediaire time steps output
		#myparam.outstep=myparam.endtime - myparam.inittime;
		myparam.outstep=3600;


	else
		myparam.levels=myinfo.levels;
		myparam.targetlevelfile=myinfo.targetlevelfile;
		myparam.outfile = myinfo.BGoutfile * ".nc"
        myparam.outstep=3600;

	end
	
	# if contains(myinfo.tidebndside,"left")
	# 	myparam.left=Bnd(2,myinfo.tidefile)
	# end

	# if contains(myinfo.tidebndside,"right")
	# 	myparam.right=Bnd(2,myinfo.tidefile)
	# end

	# if contains(myinfo.tidebndside,"top")
	# 	myparam.top=Bnd(2,myinfo.tidefile)
	# end
	# if contains(myinfo.tidebndside,"bot")
	# 	myparam.bot=Bnd(2,myinfo.tidefile)
	# end

	if !isempty(myinfo.tidepoly)

		push!(myparam.bnds,Bndseg(myinfo.tidepoly,3,myinfo.tidefile));
	end

	# Alternatively add one for the outlet

	#myparam.inittime=3600.0; # why?

	myparam.endtime=CalcEndtime(myparam, myinfo.timepicflood)

	WriteBGparam(Outfolder*outfile, myparam)

end

function CalcEndtime(BGParam::Params, peaktime::String)
	
	PT=DateTime(peaktime,"yyyy-mm-ddTHH:MM:SS")
	
	# max duration: max of flood + 24 hours 
	endtime=PT+Dates.Hour(24);

	# max duration of rain
	if !isempty(BGParam.rain)

		##

		rainfilestr=split(BGParam.rain, "?")
		println(rainfilestr[1])
		timdedata = ncread(String(rainfilestr[1]),"time")

		timeunit=ncgetatt(String(rainfilestr[1]),"time","units")

		# endtime=min(endtime,maximum(timdedata));
		timedata = CFTime.timedecode(timdedata,timeunit)
		endtime=min(endtime,timedata[end]);


	end
	# max duration of river
	for ir=1:length(BGParam.rivers)
		data = readdlm(BGParam.rivers[ir].flowfile)
		#timdedata=Float64.(data[:,1]);
		timdedata=DateTime.(data[:,1],"yyyy-mm-ddTHH:MM:SS");
		endtime=min(endtime,timdedata[end]);
	end

	#max duration of bnds

	for i=1:length(BGParam.bnds)
		# Printf.@printf(io,"bndseg = %s,%s,%d;\n",BGParam.bnds[i].polyfile,BGParam.bnds[i].file,BGParam.bnds[i].ttype);
		bndfilestr=split(BGParam.bnds[i].file, "?")
		timdedata = ncread(String(bndfilestr[1]),"time")

		timeunit=ncgetatt(String(bndfilestr[1]),"time","units")
		timedata = CFTime.timedecode(timdedata,timeunit)
		endtime=min(endtime,timedata[end]);
	end
	# if BGParam.left.ttype > 1
	# 	data = readdlm(BGParam.left.file)
	# 	timdedata=Float64.(data[:,1]);
	# 	endtime=min(endtime,maximum(timdedata));
	# end
	# if BGParam.right.ttype > 1
	# 	data = readdlm(BGParam.right.file)
	# 	timdedata=Float64.(data[:,1]);
	# 	endtime=min(endtime,maximum(timdedata));
	# end
	# if BGParam.top.ttype > 1
	# 	data = readdlm(BGParam.top.file)
	# 	timdedata=Float64.(data[:,1]);
	# 	endtime=min(endtime,maximum(timdedata));
	# end
	# if BGParam.bot.ttype > 1
	# 	data = readdlm(BGParam.bot.file)
	# 	timdedata=Float64.(data[:,1]);
	# 	endtime=min(endtime,maximum(timdedata));
	# end




	return Dates.format(endtime,"yyyy-mm-ddTHH:MM:SS")
end
#MakBGParam(ARGS[1],ARGS[2])
