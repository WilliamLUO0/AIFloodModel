using DelimitedFiles 
include("Mak_Location_File.jl")


mutable struct locstatus
	name::String
	error::Vector{String}
	started::Bool
	success::Bool
end



function readlocfile(locationfile::String)
	#
	locationraw=readdlm(locationfile,':',skipstart=5)

	locarr=locationraw[1:end-2,:]

	#duration=parse.(Int,rstrip.(locarr[:,2],','))

	locations=lstrip.(lstrip.(rstrip.(locarr[:,1],'\"'),'\t'),'\"')

	tup=rstrip.(lstrip.(strip.(rstrip.(strip.(locarr[:,2], '\t'),',')),'('),')')

	da=split.(tup,",")
	duration=parse.(Int,first.(da));
	ari=strip.(last.(da));


	return locations,duration,ari




end

function ismatch(i,s)
	
	return !(match(i,s)===nothing)
	
end

function GetGrepkline(file,regex)

	strall=""
	open(file) do f
		for i in eachline(f)
			if ismatch(regex, i) 
				strall=strall*i;
			end
		end

	end
	return strall
end

function checkstatuslocation(workflow::String, location::AbstractString,duration::Int; ARI="100", backupfolder="/nesi/nobackup/niwa03440/Cylc-Workflow-Outputs/Production_Run2")
	
	l=location;
	locstat=locstatus(location,Vector{String}(undef,0),false,false);
	locstatDS=locstatus(location,Vector{String}(undef,0),false,false);
	locstatTN=locstatus(location,Vector{String}(undef,0),false,false);

	event=ARI*"y_"*string(duration)*"h_0c"


	bg_cylc_workflow1="/home/bosserellec/cylc-run/flood_subwf_cylc-"*workflow*"/"*l*"_"*ARI*"/bg_cylc/run1/log/scheduler/log"

	
	locstat.started=isfile(bg_cylc_workflow1);

	tn_cylc_workflow1="/home/bosserellec/cylc-run/flood_subwf_cylc-"*workflow*"/"*l*"_"*ARI*"/cylc-topnet-scenario/run1/log/scheduler/log"
	locstatTN.started=isfile(tn_cylc_workflow1);


	ds_cylc_workflow1="/home/bosserellec/cylc-run/flood_subwf_cylc-"*workflow*"/"*l*"_"*ARI*"/cylc-design-storms/run1/log/scheduler/log"
	locstatDS.started=isfile(ds_cylc_workflow1);


	BGout=backupfolder*"/"*l*"/"*event*"/BG_Flood/"*l*"_"*event*"_Inundation_Floodmap_zsmax.tif"
	
	locstat.success=isfile(BGout);

	TNoutfolder=backupfolder*"/"*l*"/"*event*"/TopNet/"
	TNpattern="streamq_"
	
	if isdir(TNoutfolder)
		inTNfolder=readdir(TNoutfolder);
		locstatTN.success=any(occursin.(TNpattern,inTNfolder));
	end

	DSoutll=backupfolder*"/"*l*"/"*event*"/Design_Storm/"*event*"_latlon.nc"
	DSoutnztm=backupfolder*"/"*l*"/"*event*"/Design_Storm/"*event*"_nztm.nc"


	locstatDS.success=isfile(DSoutll) && isfile(DSoutnztm);




	logdir="/home/bosserellec/cylc-run/flood_subwf_cylc-"*workflow*"/"*l*"_"*ARI*"/bg_cylc/run1/log/scheduler/"

	logdirTN="/home/bosserellec/cylc-run/flood_subwf_cylc-"*workflow*"/"*l*"_"*ARI*"/cylc-topnet-scenario/run1/log/scheduler/"

	logdirDS="/home/bosserellec/cylc-run/flood_subwf_cylc-"*workflow*"/"*l*"_"*ARI*"/cylc-design-storms/run1/log/scheduler/"

	if(locstat.started)

		dd=readdir(logdir);
		
		for di in dd
			log=logdir*di
			open(log) do f
				for i in eachline(f)
					if ismatch(r"fail", i)
						push!(locstat.error,i)
					end
				end

			end
		end
	end
	if(locstatTN.started)

		dd=readdir(logdirTN);
		
		for di in dd
			log=logdirTN*di
			open(log) do f
				for i in eachline(f)
					if ismatch(r"fail", i)
						push!(locstatTN.error,i)
					end
				end

			end
		end
	end
	if(locstatDS.started)

		dd=readdir(logdirDS);
		
		for di in dd
			log=logdirDS*di
			open(log) do f
				for i in eachline(f)
					if ismatch(r"fail", i)
						push!(locstatDS.error,i)
					end
				end

			end
		end
	end

	return locstat,locstatTN,locstatDS
end

function checkstatus(workflow::String, locationfile::String)
	##
	# failed task will look like this:
	#		bg_cylc_Waiatoto_100 failed job:01 flows:1] did not complete required outputs: ['succeeded']
	# succedeed one look like this
	#		bg_cylc_Flaxbourne_100 running job:01 flows:1] => succeeded

	locations,durations,ari=readlocfile(locationfile);

	schedulerlog="/home/bosserellec/cylc-run/flood_subwf_cylc/"*workflow*"/log/scheduler/"

	locationstatusBG=Vector{locstatus}(undef,0);
	locationstatusTN=Vector{locstatus}(undef,0);
	locationstatusDS=Vector{locstatus}(undef,0);

	for (il, l) in enumerate(locations)

		locstat,locstatTN,locstatDS=checkstatuslocation(workflow, l,durations[il], ARI=ari[il])

		push!(locationstatusBG,locstat)
		push!(locationstatusTN,locstatTN)
		push!(locationstatusDS,locstatDS)





	end

	return locationstatusBG,locationstatusTN,locationstatusDS
end

function vecstatus(locstatus)
	loc=map(x->x.name,locstatus)
	errorstr=map(x->x.error,locstatus)
	started=map(x->x.started,locstatus)
	succ=map(x->x.success,locstatus)

	return loc,errorstr,started,succ
end

function sumstatus(loc,errorstr,started,succ; Headstr="")
	nsucc=count(succ) 

	succall=findall(succ)

	indrunning=(succ.==false) .& (started.==true) .& (isempty.(errorstr))

	indfailed=(succ.==false) .& (started.==true) .& (.!isempty.(errorstr))

	nRunning=count(indrunning)

	nfailed=count(indfailed)

	
	running=findall(indrunning)

	isfailed=findall(indfailed)

	failedother=(started.==false) .& (succ.==false)
	nfailedother = count(failedother)

	println(Headstr)
	println(length(loc)," locations; ", nsucc," Succeeded; ",nRunning," Running|need-restart; ",nfailed," failed; ", nfailedother," failed upstream|not started")



	return loc[succall],loc[running],loc[isfailed] 
end





function GetStatusSummary(workflow::String, locationfile::String; ARI="100")
	locstatusBG,locstatusTN,locstatusDS=checkstatus(workflow,locationfile);

	loc,errorstrBG,startedBG,succBG=vecstatus(locstatusBG);
	loc,errorstrTN,startedTN,succTN=vecstatus(locstatusTN);
	loc,errorstrDS,startedDS,succDS=vecstatus(locstatusDS);


	locsucBG,locrunBG,locfailBG=sumstatus(loc,errorstrBG,startedBG,succBG; Headstr="bg_cylc")
	locsucTN,locrunTN,locfailTN=sumstatus(loc,errorstrTN,startedTN,succTN; Headstr="TopNet_cylc")
	locsucDS,locrunDS,locfailDS=sumstatus(loc,errorstrDS,startedDS,succDS; Headstr="Design_Storm_cylc")

	return  locsucBG,locrunBG,locfailBG,locsucTN,locrunTN,locfailTN,locsucDS,locrunDS,locfailDS
end



function GetRunning(workflow::String, locationfile::String; ARI="100")
	locstatusb=checkstatus(workflow,locationfile,ARI=ARI);

	loc=map(x->x.name,locstatusb)
	errorstr=map(x->x.error,locstatusb)
	started=map(x->x.started,locstatusb)
	succ=map(x->x.succeeded,locstatusb)

	indRunning=(succ.==false) .& (started.==true) .& (isempty.(errorstr))

	return loc[indRunning]
end


function GetFailed(workflow::String, locationfile::String, location; ARI="100")
	locations,durations,ari=readlocfile(locationfile);
	il=findfirst(contains.(locations,location))


	locstat,locstatTN,locstatDS=checkstatuslocation(workflow, locations[il],durations[il], ARI=ari[il]);

	return locstat,locstatTN,locstatDS;

end


function Remove_lock( locationfile::String; ARI="100", backupfolder="/nesi/nobackup/niwa03440/Cylc-Workflow-Outputs/Production_Run2")
	locations,durations=readlocfile(locationfile);
	#il=findfirst(contains.(locations,location))

	
	for (il, loc) in enumerate(locations)

		for deg in ["0c" "1c" "2c" "3c"]

			event=ARI*"y_"*string(durations[il])*"h_"*deg


			lockfile=backupfolder*"/"*loc*"/"*event*"/BG_Flood/bgflood.lock"

			rm(lockfile, force=true);

			lockfile=backupfolder*"/"*loc*"/"*event*"/TopNet/topnet.lock"

			rm(lockfile, force=true);

			lockfile=backupfolder*"/"*loc*"/8m_geofabric.lock"

			rm(lockfile, force=true);

			lockfile=backupfolder*"/"*loc*"/"*event*"/Design_Storm/design_storm.lock"

			rm(lockfile, force=true);
		end
	end

end


function GetIntAfterEqual(instr)
	value=0.0;
	valuestr=first(split(lstrip(last(split(instr,'='))),' '))
	if !isempty(valuestr)
		value=parse(Float64,valuestr);
	end
	return value
end

function checkbgRuntime(workflow::String, locationfile::String; ARI="100",backupfolder="/nesi/nobackup/niwa03440/Cylc-Workflow-Outputs/Production_Run2")
	locations,durations=readlocfile(locationfile);
	#coarseRT=zeros(length(locations))
	#coarseMem=zeros(length(locations))
	#coarsenblk=zeros(length(locations))
	
	fineRT=zeros(length(locations))
	fineMem=zeros(length(locations))
	finenblk=zeros(length(locations))

	for (il, l) in enumerate(locations)
		BGlog=backupfolder*"/"*l*"/"*ARI*"y_"*string(durations[il])*"h_0c/BG_Flood/work/1/Run_BG_fine_Milan/BG_log.txt"

		if isfile(BGlog)
			println(l*" log found!")

			strgmem=GetGrepkline(BGlog,r"Model final memory usage=")
			strgrt=GetGrepkline(BGlog,r"Total runtime=")
			strgblk=GetGrepkline(BGlog,r" 0 new blocks will be created")

			fineMem[il]=GetIntAfterEqual(strgmem);
			fineRT[il]=GetIntAfterEqual(strgrt);

			if !isempty(strgblk)

				tmpblkstr=split(strgblk,' ');
				indexnblk=max(findfirst("active".==tmpblkstr)-1,1)

				finenblk[il]=parse(Int,tmpblkstr[indexnblk])
			end
		else
			println(l*" log not found")
		end
	end


	return fineRT,fineMem,finenblk
end


function scrapeResults(locationfile;backupfolder="/nesi/nobackup/niwa03440/Cylc-Workflow-Outputs/Production_Run2", collatefolder="Flow100_0c/",tag="0c")
	locations,durations,ari=readlocfile(locationfile);

	if !isdir(backupfolder*"/"*collatefolder)
		mkdir(backupfolder*"/"*collatefolder)
	end

	for ty in ["hmax", "zsmax", "humax"]
		if !isdir(backupfolder*"/"*collatefolder*"_"*ty)
			mkdir(backupfolder*"/"*collatefolder*"_"*ty)
		end
	end

	for (il,loc) in enumerate(locations)



		event=ari[il]*"y_"*string(durations[il])*"h_"*tag

		for ty in ["hmax", "zsmax", "humax"]
		 
			BG=backupfolder*"/"*loc*"/"*event*"/BG_Flood/"*loc*"_"*event*"_Inundation_Floodmap_"*ty*".tif"

			if isfile(BG)
				dest=backupfolder*"/"*collatefolder*"_"*ty*"/"*loc*"_"*event*"_Inundation_Floodmap_"*ty*".tif"

				cp(BG,dest,force=true);
			else
				@warn loc * " " *ty* " No Results found!"
			end
		end
	

	end
end



function Readflood(infile;minelev=-1.0)
	hfile=infile*"_hmax.nc"
	hgrid=ncread(hfile,"hmax")
	zsfile=infile*"_zsmax.nc"
	zsgrid=ncread(zsfile,"zsmax")
	zbgrid=zsgrid.-hgrid

	hgrid[zbgrid .< minelev].=NaN
	return hgrid
end

function Readfloodarea(infile;threshold=[0.05])
	hfile=infile*"_hmax.nc"
	xx=ncread(hfile,"x")
	dx=xx[2]-xx[1]
	hgrid=Readflood(infile)


	area=zeros(length(threshold))
	vol=zeros(length(threshold))
	for i=1:length(threshold)

		indvolold=hgrid .> threshold[i]
		area[i]=sum(indvolold)*dx*dx;
		vol[i]=sum(hgrid[indvolold])*dx*dx;
	end

	return area,vol

end

function diffFlood(loc,dur;backupfolder="/nesi/nobackup/niwa03440/Cylc-Workflow-Outputs/Production_Run2",threshold=[0.05, 0.15, 0.5, 1.0, 2.0])

	event0c="100y_" .* dur .* "h_" .* "0c"
	event1c="100y_" .* dur .* "h_" .* "1c"
	event2c="100y_" .* dur .* "h_" .* "2c"
	event3c="100y_" .* dur .* "h_" .* "3c"

	root=backupfolder .* "/" .* loc .* "/";# .* events .* "/BG_Flood/" .* loc .* "_" .* events .* "_Inundation_Floodmap"

	

	FA0c,FV0c=Readfloodarea(root .* event0c .* "/BG_Flood/" .* loc .* "_" .* event0c .* "_Inundation_Floodmap",threshold=threshold)

	FACC=zeros(3,length(threshold))
	
	FVCC=zeros(3,length(threshold))


	FACC[1,:],FVCC[1,:]=Readfloodarea(root.*event1c.* "/BG_Flood/" .* loc .* "_" .*event1c.*"_Inundation_Floodmap",threshold=threshold)
	FACC[2,:],FVCC[2,:]=Readfloodarea(root.*event2c.* "/BG_Flood/" .* loc .* "_" .*event2c.*"_Inundation_Floodmap",threshold=threshold)
	FACC[3,:],FVCC[3,:]=Readfloodarea(root.*event3c.* "/BG_Flood/" .* loc .* "_" .*event3c.*"_Inundation_Floodmap",threshold=threshold)

	
	FA0cmat=FA0c'.*ones(3)
	FV0cmat=FV0c'.*ones(3)
	FAPD=(FACC .- FA0cmat) ./ FA0cmat.*100.0
	FVPD=(FVCC .- FV0cmat) ./ FV0cmat.*100.0

	return FA0c,FV0c,FAPD,FVPD
end

function Wavepar(time,data)
	n=length(data);

	# CALCULATE ZERO DOWN-CROSSING
	w=Vector{Int64}(undef, 0)
	for i=1:n-1
	  if (data[i]>0) && (data[i+1]<=0)

	    push!(w,i);
	  end
	end

	
	Tz=0.0;
	if (length(w)>0)
	
		# MEAN ZEROS CROSING PERIOD
		Tz=(time[w[end]]-time[w[1]])/(length(w)-1);
	end
	
	Hrms=sqrt(sum(data.^2)/n)

	return Tz,Hrms

end



function Checktide(infile)

	time=ncread(infile,"time")

	h=dropdims(ncread(infile,"h_P0",start=[1,1,1], count = [-1,-1,1]),dims=3)

	h[h.>9000.0].=-1;

	maxxloc=argmax(h)

	zsTS=ncread(infile,"zs_P0",start=[maxxloc[1],maxxloc[2],1], count = [1,1,-1])

	zsTS=dropdims(ncread(infile,"zs_P0",start=[maxxloc[1],maxxloc[2],1], count = [1,1,-1]),dims=(1,2))

	zsTS=zsTS .- (sum(zsTS) / length(zsTS))

	Tz,Hrms=Wavepar(time,zsTS);

	tidegood=false

	if( (Tz/3600.0 > 10.0) .& (Tz/3600.0 .< 16.0) .& ( Hrms .> 0.2) .& (Hrms .< 1.5)) 
		tidegood=true
	end

	return tidegood
end

function CheckBGSummary(gpkgfile; outcsv="",backupfolder="/nesi/nobackup/niwa03440/Cylc-Workflow-Outputs/Production_Run2", checkreportroot="Checkreport")
	durations=[6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72];
	ari=[20, 50, 100]
	climatechange=[0,1,2,3]

	ccarray=Int64.(climatechange .* ones(length(durations))')
	durationarray=Int64.(ones(length(climatechange)) .* durations')

	events="100y_" .* string.(durationarray) .* "h_" .* string.(ccarray) .* "c"

	presentday="100y_" .* string.(durationarray) .* "h_" .* string.(ccarray) .* "c"

	savecsv=!isempty(outcsv)

	if savecsv
		open(outcsv,"w") do io
			Printf.@printf(io,"Location,duration,rainARI4flow,")
			Printf.@printf(io,"perc_complete,tidecheck,")
			Printf.@printf(io,"area_flooded_100ARI_rain,")
			Printf.@printf(io,"perc_area_increase_w_1degc,perc_area_increase_w_2degc,perc_area_increase_w_3degc\n")

			Printf.@printf(io,"units:,h,year,")
			Printf.@printf(io,"%%,0:notgood 1:good,")
			Printf.@printf(io,"km^2,")
			Printf.@printf(io,"%%,%%,%%\n")
		end
	end

	names,EvDur,EVARID,QEVA=GetDurations(gpkgfile, durations=durations)

	totalcount=0

	locdone=0

	for (il,loc) in enumerate(names)

		println(loc)

		

		alldirs=backupfolder .* "/" .* loc .* "/" .* events .* "/BG_Flood/" .* loc .* "_" .* events .* "_Inundation_Floodmap_hmax.nc"
		mainfolderall=backupfolder .* "/" .* loc .* "/" .* events .* "/BG_Flood/"
		

		isdone=findall(isfile.(alldirs))

		ispresent=(findall(isfile.(alldirs[1,:])))

		totalcount=totalcount+count(isfile.(alldirs))

		rainari4flow=-1;


		FA0c=NaN;
		FV0c=NaN;
		FAPD=zeros(3).*NaN;
		FVPD=zeros(3).*NaN;


		dur=-1;

		tidecheck=false


		if(any(isfile.(alldirs)))
			# Check the tide at the deepest location
			mainBGout=mainfolderall[isdone[1]] .* loc .* "_" .* events[isdone[1]] .* "_Inundation.nc"

			# mainElevation=mainfolderall[1,ispresent][1] .* loc .* "_" .* events[1,ispresent][1] .* "_Inundation.nc"
			tidecheck=Checktide(mainBGout);


			dur=durations[argmax(dropdims(sum(isfile.(alldirs),dims=1),dims=1))]


			# Calculate the area flooded and % increase with Climate change for different depth threshold

			if(all(dropdims(sum(isfile.(alldirs),dims=2),dims=2).>0))
				FA0cv,FV0cv,FAPDm,FVPDm=diffFlood(loc,string(dur),threshold=[0.5])
				FA0c=FA0cv[1];
				FV0c=FV0cv[1];
				FAPD[:]=FAPDm[:,1];
				FVPD[:]=FVPDm[:,1];
			end
		
		end

		if savecsv
			open(outcsv,"a") do io

				Printf.@printf(io,"%s,%d,%d,",loc,dur,rainari4flow)
				Printf.@printf(io,"%d,%d,",count(isfile.(alldirs))/4*100,tidecheck)
				Printf.@printf(io,"%2.2f,",FA0c/(1000*1000))
				Printf.@printf(io,"%d,%d,%d\n",FAPD[1],FAPD[2],FAPD[3])
			
			end
		end

	end

	println(totalcount / (length(names)*4) *100)
end



		






		






