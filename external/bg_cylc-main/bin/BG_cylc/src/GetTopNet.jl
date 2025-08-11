
using NetCDF, Printf, Dates, DelimitedFiles

function Nearestreach(in,vectx,vecty,areaup)
	distall=hypot.(in[1].-vectx,in[2].-vecty)
	id=argmin(distall)

	idsort = sortperm(distall)
	ids=1
	areamax = areaup[idsort[1]]
	for i=1:20
		if (areaup[idsort[i]]>areamax)
			ids=i;
			areamax=areaup[idsort[i]]
		end
	end

	id=idsort[ids]
	return id
end


function getreadindex(file,rchid)
	rchidall=ncread(file,"rchid")

	id=findfirst(rchidall.==rchid)
	return id
end

function getreachidindex(allfile::Vector{String},rchid)
	fid="";
	#while isnothing(id)
	for ifile=1:length(allfile)
		file=allfile[ifile];
		#println(file)
		id=getreadindex(file,rchid)
		if !isnothing(id)
			fid=file
			break;
		end
	end
	return fid
end


function GetFlow(file,id,ens;varname="river_flow_rate_mod")

	#println(id)
	#check the number of dimention for the variable

	finfo=NetCDF.open(file);

	v=finfo[varname];

	ndims=length(size(v));

	if ndims <= 2
		start=[id,1]
		count=[1,-1]
	elseif ndims == 3
		start=[ens,id,1]
		count=[1,1,-1]
	elseif ndims == 4
		start=[ens,1,id,1]
		count=[1,1,1,-1]
	end
	
	flowraw=ncread(file,varname,start=start,count=count)

	flowraw=dropdims(flowraw,dims=tuple(findall(size(flowraw).==1)...))
	flowraw[flowraw.<0.0].=0.0

	return flowraw
end

function GetRain(file,id,ens;varname="aprecip")

	#println(id)
   flowraw=dropdims(ncread(file,varname,start=[ens,1,id,1],count=[1,1,1,-1]),dims=(1,2,3))
   flowraw[flowraw.<0.0].=0.0

   return flowraw
end

function GetMedianFlow(file,id)
#flowraw=dropdims(ncread(file,"median_river_flow_rate_mod",start=[id,1],count=[1,-1]),dims=(1))
#	flowraw[flowraw.<0.0].=0.0

	Meanflow=GetFlow(file,id,1).*0.0;

	for ens=1:1
		flow=GetFlow(file,id,ens)
		Meanflow.= Meanflow .+flow
	end

	Meanflow .= Meanflow./1;

	return Meanflow
end

function GetHighFlow(file,id)
#flowraw=dropdims(ncread(file,"ptile95_river_flow_rate_mod",start=[id,1],count=[1,-1]),dims=(1))
#	flowraw[flowraw.<0.0].=0.0

	Maxflow=GetFlow(file,id,1).*0.0;

	for ens=1:1
		flow=GetFlow(file,id,ens)
		Maxflow.= max.(Maxflow ,flow)
	end



	return Maxflow

end

function GetTopNetFlow(file,rchid,BGreftime;varname="river_flow_rate_mod",israin=false)
	timeref=ncgetatt(file,"time", "units")
	refstr=string(last(split(timeref,"since")))
	refunit=string(first(split(timeref,"since")))
	DT_ref=DateTime(strip(refstr),"yyyy-mm-dd HH:00")


	if contains(lowercase(refunit),lowercase("hours"))
		timeraw=Dates.Hour.(ncread(file,"time")).+DT_ref
	else
		timeraw=Dates.Second.(ncread(file,"time")).+DT_ref
	end
	#BGreftime=DateTime(2023,02,12,12,0,0);

	timesec=Dates.Second.(timeraw .- BGreftime)

	BGtimesec=map((x) -> x.value,timesec);

	#lon=ncread(file,"start_lon")
	#lat=ncread(file,"start_lat")

	#areaupstream=ncread(file,"uparea")

	#id=Nearestreach(loc,lon,lat,areaupstream)
	id=getreadindex(file,rchid)
	if !israin
		Flow=GetFlow(file,id,1,varname=varname)
	else
		Flow=GetRain(file,id,1,varname=varname)
	end
	#Flowhigh=GetHighFlow(file,id)

	#maxflow=maximum(Flow)

	# open(rootname*"_median.txt","w") do io
    #  for ddii=1:length(timeraw)
    #      Printf.@printf(io,"%s\t%f\n",Dates.format(timeraw[ddii],"yyyy-mm-ddTHH:MM:SS"),Flowmed[ddii]);

    #  end

	# end

	# open(rootname*"_median_BG.txt","w") do io
    #  for ddii=1:length(timeraw)
    #      Printf.@printf(io,"%f\t%f\n",BGtimesec[ddii],Flowmed[ddii]);

    #  end

	# end
	#println("reachid= $rchid; ID = $id; maxflow = $maxflow");


	

	return timeraw,BGtimesec,Flow
end


function GetInjectionXY(TNfolder,injectionfile;outfolder="",outfile="Injection_XY.txt")
	
	reachids=Vector{Int64}(undef,0);
	injectfile=Vector{String}(undef,0);
	StremQfiles=Vector{String}(undef,0);
	ids=Vector{Int}(undef,0);
	lon=Vector{Float64}(undef,0);
	lat=Vector{Float64}(undef,0);
	
	try
		injection=readdlm(injectionfile, ',');


		reachids=Int.(injection[:,1]);

		listfile=readdir(TNfolder);

		StremQfiles=TNfolder.*listfile[contains.(listfile,"streamq")];

		injectfile=Vector{String}(undef,length(reachids));

		ids=Vector{Int}(undef,length(reachids));

		lon=Vector{Float64}(undef,length(reachids));
		lat=Vector{Float64}(undef,length(reachids));
	catch 
		reachids=Vector{Int64}(undef,0);
	end



	for i=1:length(reachids)
		injectfile[i]=getreachidindex(StremQfiles,reachids[i]);

		if !isempty(injectfile[i])
			
		

			ids[i]=getreadindex(injectfile[i],reachids[i])

			lonall=ncread(injectfile[i],"end_lon")
			latall=ncread(injectfile[i],"end_lat")

			lon[i]=lonall[ids[i]]
			lat[i]=latall[ids[i]]
		else
			println("Injection reachid not found in TopNet; rid="*string(reachids[i]))
		end


	end

	

	open(outfolder*outfile,"w") do io
		for i=1:length(reachids)
			if !isempty(injectfile[i])
				Printf.@printf(io,"%d,%f,%f,%d\n",reachids[i],lon[i],lat[i],reachids[i])
			end
		end
	end

end



"""
    GetTopNetFlow(TNfile,injectionfile)

TBW
"""
function GetTopNetFlow(TNfolder,injectionfile; outfolder="", reftime="")
	#ngaruroroll=(176.72872,-39.59423)
	#Tutaekurill=(176.78795,-39.51189)

	#Eskll=(176.84692,-39.38997)
	#Tukitukill=(176.92776,-39.69111)

	#Maraetotarall=(176.98615,-39.70313)
	#Ngaruroro=8207055;
	#Tutaekuri=8202391;
	#Esk=8194101;
	#Tukituki=8213911;
	#Maraetotara=8213000;

	# List all streamQ files in the TN folder
	listfile=readdir(TNfolder);

	StremQfiles=TNfolder.*listfile[contains.(listfile,"streamq")];
	

	BGreftime=DateTime(2020,01,01,0,0,0);

	if isempty(reftime)
		timeref=ncgetatt(StremQfiles[1],"time", "units")
		BGreftime=DateTime(replace(timeref, "hours since " => ""),"yyyy-mm-dd HH:00")
	else
		BGreftime=DateTime(reftime,"yyyy-mm-dd HH:MM:SS")
	end

	reachids=Vector{Int64}(undef,0);
	#injectfile=Vector{String}(undef,0);
	#StremQfiles=Vector{String}(undef,0);
	#ids=Vector{Int}(undef,0);
	#lon=Vector{Float64}(undef,0);
	#lat=Vector{Float64}(undef,0);
	try
		injection=readdlm(injectionfile, ',');

		reachids=Int.(injection[:,1]);

		xloc=Float64.(injection[:,2])
		yloc=Float64.(injection[:,3])

		Name=injection[:,4]

		injectfile=Vector{String}(undef,length(reachids));

		for i=1:length(reachids)
			injectfile[i]=getreachidindex(StremQfiles,reachids[i]);
		end


		open(outfolder*"BG_river_bnd.txt","w") do io
			for i=1:length(reachids)
				if !isempty(injectfile[i])
					println(string(Name[i]))
					Namei=Printf.@sprintf("%s",Name[i]);
					timeraw,BGtimesec,Flow=GetTopNetFlow(injectfile[i],reachids[i],BGreftime)

					maxflow=maximum(Flow);

					open(outfolder*Namei*".txt","w") do io
						for ddii=1:length(timeraw)
							Printf.@printf(io,"%s\t%f\n",Dates.format(timeraw[ddii],"yyyy-mm-ddTHH:MM:SS"),Flow[ddii]);
				
						end
				
					end
				
					open(outfolder*Namei*"_BG.txt","w") do io
						for ddii=1:length(timeraw)
							Printf.@printf(io,"%f\t%f\n",BGtimesec[ddii],Flow[ddii]);
				
						end
				
					end




					riverfootprint=max(sqrt(maxflow/0.5)/2,10)

					Printf.@printf(io,"%s,%f,%f,%f,%f;\n","river = " * Namei * "_BG.txt",xloc[i]-riverfootprint,xloc[i]+riverfootprint,yloc[i]-riverfootprint,yloc[i]+riverfootprint);
				end
			end
		end
	catch
		@error "No Injection point found!"
	
	end	

	#GetTopNetFlow(TNfile,Maraetotara,"Maraetotara")
	#GetTopNetFlow(TNfile,Tukituki,"Tukituki")
	#GetTopNetFlow(TNfile,Esk,"Esk")
	#GetTopNetFlow(TNfile,Tutaekuri,"Tutaekuri")
	#GetTopNetFlow(TNfile,Ngaruroro,"Ngaruroro")
end

function GetTopNetPeakref(TNfolder,ocenreachidfile,outfile)
	

	
	oceanrids=readdlm(ocenreachidfile, ',');
	listfile=readdir(TNfolder);

	StremQfiles=TNfolder.*listfile[contains.(listfile,"streamq")];
	BGreftime=DateTime(2020,01,01,0,0,0);

	Qfile=Vector{String}(undef,length(oceanrids));

	maxQ=Vector{Float64}(undef,length(oceanrids));
	maxid=Vector{Int64}(undef,length(oceanrids));

	for i=1:length(oceanrids)
		Qfile[i]=getreachidindex(StremQfiles,oceanrids[i]);
	end

	for i=1:length(oceanrids)
		if !isempty(Qfile[i])
			#println(string(Name[i]))
			#Namei=Printf.@sprintf("%s",Name[i]);
			timeraw,BGtimesec,Flow=GetTopNetFlow(Qfile[i],oceanrids[i],BGreftime)

			maxflowid=argmax(Flow);
			maxQ[i] = Flow[maxflowid];
			maxid[i] = maxflowid;
		end
	end

	

	reachidbig=argmax(maxQ)

	timeraw,BGtimesec,Flow=GetTopNetFlow(Qfile[reachidbig],oceanrids[reachidbig],BGreftime)
	timeref=timeraw[maxid[reachidbig]];
	BGtimeref=timeraw[1]-Dates.Hour(1);

	println(Dates.format(timeref,"yyyy-mm-ddTHH:MM:SS"))

	open(outfile,"w") do io
		Printf.@printf(io,"%s\n",Dates.format(timeref,"yyyy-mm-ddTHH:MM:SS"))
		Printf.@printf(io,"%s\n",Dates.format(BGtimeref,"yyyy-mm-dd HH:MM:SS"))
		Printf.@printf(io,"%s\n",Dates.format(BGtimeref,"yyyy-mm-ddTHH:MM:SS"))
		Printf.@printf(io,"%s\n",Dates.format(BGtimeref+Dates.Hour(1),"yyyy-mm-ddTHH:MM:SS"))
	end
end





#GetTopNetFlowNapier(ARGS[1],ARGS[2])
