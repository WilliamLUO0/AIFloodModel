using ArchGDAL, Printf

include("GetTopNet.jl")

using CurveFit

function GetDurations(inputfile;durations=[6, 12, 24, 48, 72], getgauge=false)
    dataset = ArchGDAL.read(inputfile);
    nlayer=ArchGDAL.nlayer(dataset)
    basinlayerid=0
    for l=0:(nlayer-1)
        tmplayer=ArchGDAL.getlayer(dataset, l)
        name=ArchGDAL.getname(tmplayer);
        isbasin=occursin(lowercase(name),"basin")
        if isbasin
            basinlayerid=l 
        end
    end


    basin = ArchGDAL.getlayer(dataset, basinlayerid)
    nfeat = ArchGDAL.nfeature(basin)

    featuredefn =ArchGDAL.layerdefn(basin)
    nfield=ArchGDAL.nfield(featuredefn)

    fieldname2get=["name","tocs","eva_rec2s","flood_stats_rec1s","flood_stats_sites"]
    fieldid2get=zeros(Int64,size(fieldname2get))


    for i=0:(nfield-1)
        fielddefn=ArchGDAL.getfielddefn(featuredefn, i)
        fieldname=ArchGDAL.getname(fielddefn)
        checkname=fieldname2get.==fieldname

        if any(checkname)
            fieldid2get[findfirst(checkname)]=i
        end
    end






    names=Vector{String}(undef,0)
    EvDur=Vector{Int64}(undef,0)
    EVARID=Vector{Int64}(undef,0)
    QEVA=Vector{Float64}(undef,0)

    for i=1:(nfeat) # should that be 0 to nfeat-1?
		#try
        ArchGDAL.getfeature(basin, i) do feature
            name = ArchGDAL.getfield(feature, fieldid2get[1]) 
            val = ArchGDAL.getfield(feature, fieldid2get[2]) 
            evaridstr = ArchGDAL.getfield(feature, fieldid2get[3]) 
            if !getgauge

                
                qevastr = ArchGDAL.getfield(feature, fieldid2get[4]) 
            else
                qevastr = ArchGDAL.getfield(feature, fieldid2get[5]) 
            end
            alltocstr=replace(val, "NULL"=>"-999.0")
            #replace("GFG Geeks.", "GFG" => "GeeksforGeeks"
            alltocs=parse.(Float64,split(alltocstr, ","))


            allqevastr=split(qevastr, ",");
            allqevastr[isempty.(allqevastr)].="-999.9"
            allevarid=parse.(Int64,split(evaridstr, ","))
            allqeva=parse.(Float64,allqevastr)

            indexlargestQ=argmax(allqeva)

            toc=maximum(alltocs)

            durationGTtoc=durations.>(2*toc)

            duri=1
            if any(durationGTtoc)
                duri=findfirst(durations.>(2*toc))
            else
                duri=length(durations)# i.e. no duration GT 2*TOC then use the largest duration 
            end

            dur=durations[duri]


            
            validname=
            try parse(Int64,name)
                "missing"
            catch e
                name
            end
            
            
            ismiss=occursin("missing", validname)
            isQok=allqeva[indexlargestQ]>0.0;
            if !ismiss & isQok
                #println(validname,", ",dur)
                push!(names, validname);
                push!(EvDur,dur);
                push!(EVARID,allevarid[indexlargestQ])
                push!(QEVA,allqeva[indexlargestQ])
            else
                @warn "Skipping feature "*name*" / "*validname*" i="*string(i)*"; has a bad name or no data in qevastr="*qevastr*" Q="*string(allqeva[indexlargestQ])
            end
            
            
        end
		# catch e
        #     name="dummy"
        #     ArchGDAL.getfeature(basin, i) do feature
        #         name = ArchGDAL.getfield(feature, fieldid2get[1]) 
        #     end
		# 		@warn "problem with reading the file "*inputfile*" at feature "*name*" i="*string(i)
		# end
    end

    return names,EvDur,EVARID,QEVA
    
end
function writelocfile(names::Vector{String},EvDur::Vector{Int64};outfile="locations.j2")
    open(outfile,"w") do io
        Printf.@printf(io,"#!Jinja2\n\n")
        Printf.@printf(io,"{# LOCATION: DURATION #}\n")
        Printf.@printf(io,"{%% set LOCATIONS =\n\t{\n")
        
        for i=1:length(names)

            Printf.@printf(io,"\t\t\"%s\": %d,\n",names[i],EvDur[i])
        end
        Printf.@printf(io,"\t}")
        Printf.@printf(io,"\n%%}\n")
            
    

    end
end

function writelocfile(names::Vector{String},EvDur::Vector{Int64},EVA::Vector{Int64};outfile="locations_EVA.j2")
    open(outfile,"w") do io
        Printf.@printf(io,"#!Jinja2\n\n")
        Printf.@printf(io,"{# LOCATION: DURATION: RETURN_PERIOD #}\n")
        Printf.@printf(io,"{%% set LOCATIONS =\n\t{\n")
        
        for i=1:length(names)

            Printf.@printf(io,"\t\t\"%s\": (%d , %d),\n",names[i],EvDur[i],EVA[i])
        end
        Printf.@printf(io,"\t}")
        Printf.@printf(io,"\n%%}\n")
            
    

    end
end


function MakeLocFile(gpkgfile;durations=[6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72],ARI=100,outfile="locations.j2")
    names,EvDur,EVARID,QEVA=GetDurations(gpkgfile, durations=durations)

    EvARI=Int64.(ones(length(names)) .* ARI)
    writelocfile(names,EvDur,EvARI,outfile=outfile)
end




function interpEVA(ari,q,refq)

    lari=log10.(ari)

    fit=curve_fit(LinearFit,q, lari)



    

    
    #prev + (time) / (timenext)*(next - prev);
    #lariref= lari[a1] + (refq-q[a1])/(q[a2]-q[a1])*(lari[a2]-lari[a1]);


    ariref=10^(fit(refq));
    
    return ariref
end



function GetARI(loc,Dur,rid;EVAref=[20, 50, 100],backupfolder="/nesi/nobackup/niwa03440/Cylc-Workflow-Outputs/Initial-Runs/")
    qARI=zeros(length(EVAref))
    #RP=Vector{Float64}(undef,0)
    folderEVA=string.(EVAref) .* "y_" .* string(Dur) .* "h_0c"

    for (ia,ARI) in enumerate(EVAref)

        TNfolder=backupfolder * "/" * loc * "/" * folderEVA[ia] * "/TopNet/"
        # Find the file that contain the right reachid
        listfile=readdir(TNfolder);

        StremQfiles=TNfolder.*listfile[contains.(listfile,"streamq")];

        thatfile = getreachidindex(StremQfiles,rid)

        # read peak streamq 

        if !isempty(thatfile)
            timeref=ncgetatt(thatfile,"time", "units")
            BGreftime=DateTime(replace(timeref, "hours since " => ""),"yyyy-mm-dd HH:00")
            time,BGtime,q=GetTopNetFlow(thatfile,rid,BGreftime)
            maxq=maximum(q);
            qARI[ia]=maxq
            #push!(qARI,maxq)
            #push!(RP,ARI)
        end
    end
    
    return EVAref,qARI
end

function GetTNTS(loc,Dur,rid,EVAref; backupfolder="/nesi/nobackup/niwa03440/Cylc-Workflow-Outputs/Production_Run1",qfile="streamq",varname="river_flow_rate_mod",israin=false)
    #
    folderEVA=string.(EVAref) .* "y_" .* string(Dur) .* "h_0c"
    
    TNfolder=backupfolder * "/" * loc * "/" * folderEVA * "/TopNet/"
    # Find the file that contain the right reachid
    listfile=readdir(TNfolder);

    StremQfiles=TNfolder.*listfile[contains.(listfile,qfile)];

    thatfile = getreachidindex(StremQfiles,rid)

    

    # read peak streamq 

    time=[];
    q=[];

    if !isempty(thatfile)
        timeref=ncgetatt(thatfile,"time", "units")

        refstr=string(last(split(timeref,"since")))
        BGreftime=DateTime(strip(refstr),"yyyy-mm-dd HH:00")
        
        
        
        time,BGtime,q=GetTopNetFlow(thatfile,rid,BGreftime,varname=varname,israin=israin)
    end
    
    return time,q
end



function MakeEVAloc(gpkgfile::String;durations=[6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72],outfile="locations_EVA.j2",EVAref=[20, 50, 100],skipdone=true,backupfolder="/nesi/nobackup/niwa03440/Cylc-Workflow-Outputs/Initial-Runs/",loctag="",onlythese=String[],getgauge=false)
    
    names,EvDur,EVARID,QEVA=GetDurations(gpkgfile, durations=durations,getgauge=getgauge)

    if (!isempty(onlythese))
        #
        actuallistid=Vector{Int64}()
        for (il,oname) in enumerate(onlythese)
            
            
            
            itsthere=any(oname.==names);
            if itsthere
                push!(actuallistid,findfirst(oname.==names))
            else
                @warn "Location not in geopackage: "*oname;
            end
        end

        names=names[actuallistid];
        EvDur=EvDur[actuallistid];
        EVARID=EVARID[actuallistid];
        QEVA=QEVA[actuallistid];




    end

    RainEVA4Q=Vector{Float64}(undef,length(names))

    isalreadydone=isempty.(names);

    fill!(RainEVA4Q,0.0);


    for (il,loc) in enumerate(names)
        # Check if all 3 EVAref have been done 

        folderEVA=string.(EVAref) .* "y_" .* string.(durations') .* "h_0c"

        isready=all(maximum(isdir.(backupfolder .* "/" .* loc .* loctag .* "/" .* folderEVA),dims=2));

        

        

        if isready
            println(loc);

            maxqdur=zeros(size(durations))
            maxRPdur=zeros(size(durations))

            for (idur,dur) in enumerate(durations)
                try
                     
                    RP,qARI=GetARI(loc* loctag,dur,EVARID[il],EVAref=EVAref,backupfolder=backupfolder)

                


                    if length(qARI)>1

                        maxqdur[idur]=last(qARI);
                        
                        maxRPdur[idur]=interpEVA(RP,qARI,QEVA[il])
                        
                    end
                catch
                    println("Issues with ", loc, " and duration = ",  dur )
                end

                # Calc Rain EVA cooresponding to floodEVA
                # using linear interpolation 
                
            end

            println(loc * " TargetQ =", QEVA[il]," Q:", maxqdur," RP:",maxRPdur)

            twoxTOC=EvDur[il];
            fourxTOC=2*twoxTOC;

            durationvalidids=(durations .>= twoxTOC) .& (durations .<= fourxTOC)

            maxqdur[.!durationvalidids].=-1.0;

            maxid=argmax(maxqdur)
            if (!isnan(maxRPdur[maxid]) )
                println(loc * " is ready! DUR = ",durations[maxid]/(0.5*EvDur[il])," xTOC; ARI = ",maxRPdur[maxid],"y ([2-250] used)")

                

                RainEVA4Q[il] = min(max(maxRPdur[maxid],2.0),250);
                EvDur[il] = durations[maxid]

                event=string(Int64(round(min(maxRPdur[maxid],2000.0))))*"y_"*string(durations[maxid])*"h_0c"

                BGout=backupfolder*loc*"/"*event*"/BG_Flood/"*loc*"_"*event*"_Inundation_Floodmap_zsmax.tif"
	
	            if (isfile(BGout) & skipdone);
                    isalreadydone[il]=true
                end
            end



            

        end
    end

    indexisready = .!(isnan.(RainEVA4Q)) .& .!(isalreadydone) 
   

    writelocfile(names[indexisready],EvDur[indexisready],Int64.(round.(RainEVA4Q[indexisready])),outfile=outfile);
end



function MakeLocRainFile(gpkgfile,ARI;durations=[6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72],outfile="locations_Rain100.j2",EVAref=[20, 50, 100],skipdone=true,backupfolder="/nesi/nobackup/niwa03440/Cylc-Workflow-Outputs/Initial-Runs/",loctag="",onlythese=String[],getgauge=false)
    
    names,EvDur,EVARID,QEVA=GetDurations(gpkgfile, durations=durations,getgauge=getgauge)

   
    

    if (!isempty(onlythese))
        #
        actuallistid=Vector{Int64}()
        for (il,oname) in enumerate(onlythese)
            
            
            
            itsthere=any(oname.==names);
            if itsthere
                push!(actuallistid,findfirst(oname.==names))
            else
                @warn "Location not in geopackage: "*oname;
            end
        end

        names=names[actuallistid];
        EvDur=EvDur[actuallistid];
        EVARID=EVARID[actuallistid];
        QEVA=QEVA[actuallistid];




    end
    
    isalreadydone=isempty.(names);

    for (il,loc) in enumerate(names)
        # Check if all 3 EVAref have been done 

        folderEVA=string.(EVAref) .* "y_" .* string.(durations') .* "h_0c"

        isready=all(isdir.(backupfolder .* "/" .* loc .* loctag .* "/" .* folderEVA));

        

        

        if isready
            println(loc);

            maxqdur=zeros(size(durations))
            #maxRPdur=zeros(size(durations))

            for (idur,dur) in enumerate(durations)
                try
                     
                    RP,qARI=GetARI(loc* loctag,dur,EVARID[il],EVAref=EVAref,backupfolder=backupfolder)

                


                    

                    maxqdur[idur]=last(qARI);
                        
                   
                
                catch
                    println("Issues with ", loc, " and duration = ",  dur )
                end

                # Calc Rain EVA cooresponding to floodEVA
                # using linear interpolation 
                
            end

            #println(loc * " TargetQ =", QEVA[il]," Q:", maxqdur," RP:",maxRPdur)

            #maxid=argmax(maxqdur)
            if (!any(maxqdur .==0.0))
                

                twoxTOC=EvDur[il];
                fourxTOC=2*twoxTOC;

                durationvalidids=(durations .>= twoxTOC) .& (durations .<= fourxTOC)

                maxqdur[.!durationvalidids].=-1.0;

                maxid=argmax(maxqdur)
                println(loc * " is ready! DUR = ",durations[maxid]/(0.5*EvDur[il])," xTOC ")

                #RainEVA4Q[il] = maxRPdur[maxid]
                EvDur[il] = durations[maxid]

            end

            event=string(Int64(round(ARI)))*"y_"*string(EvDur[il])*"h_0c"

            BGout=backupfolder*loc*"/"*event*"/BG_Flood/"*loc*"_"*event*"_Inundation_Floodmap_zsmax.tif"

            if (isfile(BGout) & skipdone);
                isalreadydone[il]=true

                println(loc * " with " * event * "is done! Skipping." )
            end
        



            

        end
    end

    indexisready = .!(isalreadydone) 
   

    writelocfile(names[indexisready],EvDur[indexisready],Int64.(round.(EvDur[indexisready].*0.0.+ARI)),outfile=outfile);
end


function CompareFloodAnalysiswithTopNet(Floodfile;backupfolder="/nesi/nobackup/niwa03440/Cylc-Workflow-Outputs/Production_Run1",EVAref=[20, 50, 100],durations=[6, 12, 18, 24, 36, 42, 48, 54, 60, 66, 72],outfile="FlowARICheck.txt")

    #Read the geopackage file
    #names,EvDur,EVARID,QEVA=GetDurations(gpkgfile, durations=durations,getgauge=getgauge)

    
    # Read the Flood stats data
    
    dataset = ArchGDAL.read(Floodfile);
    layer=ArchGDAL.getlayer(dataset,0)

    nfeat = ArchGDAL.nfeature(layer);

    BasinName=Vector{String}(undef,0)
    reachid=Vector{Int64}(undef,0)
    
    Qg100=Vector{Float64}(undef,0)
    Qg50=Vector{Float64}(undef,0)
    Qg20=Vector{Float64}(undef,0)
    Qhc100=Vector{Float64}(undef,0)
    Qhc50=Vector{Float64}(undef,0)
    Qhc20=Vector{Float64}(undef,0)
    nztmx=Vector{Int64}(undef,0)
    nztmy=Vector{Int64}(undef,0)
    isEVA=Vector{Bool}(undef,0)

    for i=1:(nfeat)
		
        ArchGDAL.getfeature(layer, i) do feature
            name = ArchGDAL.getfield(feature, 39) # WARNING HARD WIRED
            dn2rid = ArchGDAL.getfield(feature, 42) # WARNING HARD WIRED
            
            qr100=ArchGDAL.getfield(feature, 26)
            qr50=ArchGDAL.getfield(feature, 25)
            qr20=ArchGDAL.getfield(feature, 24)

            qh20=ArchGDAL.getfield(feature,48)
            qh50=ArchGDAL.getfield(feature,49)
            qh100=ArchGDAL.getfield(feature,50)

            xx=ArchGDAL.getfield(feature, 3)
            yy=ArchGDAL.getfield(feature, 4)

            isevaq=ArchGDAL.getfield(feature,59)

            isok = !any([ismissing(name), ismissing(dn2rid),ismissing(qr100),ismissing(qr50), ismissing(qr20)])

            if isok
                push!(BasinName,name)
                push!(reachid,dn2rid)
                
                push!(Qg100,qr100)
                push!(Qg50,qr50)
                push!(Qg20,qr20)
                push!(Qhc100,qh100)
                push!(Qhc50,qh50)
                push!(Qhc20,qh20)
                push!(nztmx,xx)
                push!(nztmy,yy)
                push!(isEVA,isevaq)
            end
        end
    end 

    QTN100=zeros(size(Qg100));
    QTN50=zeros(size(Qg100));
    QTN20=zeros(size(Qg100));
    Durqmax=zeros(size(Qg100));

    # Read Henderson and Collins data


    # Read the topnet results only select the durations that produces the highest peak
    for il=1:length(Qg100)
        maxqdur=zeros(size(EVAref))
        durrC=0.0
        for (id,dur) in enumerate(durations)
            rint=EVAref
            q=zeros(size(EVAref))
            try
            
                rint,q=GetARI(BasinName[il],dur,reachid[il],EVAref=EVAref,backupfolder=backupfolder)
            catch
                @warn "Issue here: BasinName[il],dur,reachid[il]"
            end
            if q[3]>maxqdur[3]
                durrC=dur;
            end
            maxqdur=max.(maxqdur,q);
        end
        QTN100[il]=maxqdur[3];
        QTN50[il]=maxqdur[2];
        QTN20[il]=maxqdur[1];
        Durqmax[il]=durrC;
    end
    
    # Output to a readable file
    open(outfile,"w") do io
        Printf.@printf(io,"nztmx,nztmy,dn2reachid,isEVA,Basin,durationmax,Q100FS,Q100HC,Q100TN,Q50FS,Q50HC,Q50TN,Q20FS,Q20HC,Q20TN\n")
        for il=1:length(Qg100)
            Printf.@printf(io,"%f,%f,%d,%d,%s,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\n",nztmx[il],nztmy[il],reachid[il],isEVA[il],BasinName[il],Durqmax[il],Qg100[il],Qhc100[il],QTN100[il],Qg50[il],Qhc50[il],QTN50[il],Qg20[il],Qhc20[il],QTN20[il]);
   
        end
   
    end
   




end

function CompareFloodAnalysiswithTopNet4Sam(Floodfile;backupfolder="/nesi/nobackup/niwa03440/Cylc-Workflow-Outputs/Production_Run1",EVAref=[100],durations=[6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72],outfile="FlowARICheck_30-06-2025.txt")

    #Read the geopackage file
    #names,EvDur,EVARID,QEVA=GetDurations(gpkgfile, durations=durations,getgauge=getgauge)

    
    # Read the Flood stats data
    
    dataset = ArchGDAL.read(Floodfile);
    layer=ArchGDAL.getlayer(dataset,0)

    nfeat = ArchGDAL.nfeature(layer);

    BasinName=Vector{String}(undef,0)
    reachid=Vector{Int64}(undef,0)
    
    Qg100=Vector{Float64}(undef,0)
    #Qg50=Vector{Float64}(undef,0)
    # Qg20=Vector{Float64}(undef,0)
    Qhc100=Vector{Float64}(undef,0)
    # Qhc50=Vector{Float64}(undef,0)
    # Qhc20=Vector{Float64}(undef,0)
    nztmx=Vector{Int64}(undef,0)
    nztmy=Vector{Int64}(undef,0)
    isEVA=Vector{Bool}(undef,0)
    toc=Vector{Float64}(undef,0)
    accarea=Vector{Float64}(undef,0)

    for i=1:(nfeat)
		
        ArchGDAL.getfeature(layer, i) do feature
            name = ArchGDAL.getfield(feature, 39) # WARNING HARD WIRED
            dn2rid = ArchGDAL.getfield(feature, 42) # WARNING HARD WIRED
            
            qr100=ArchGDAL.getfield(feature, 26)
            # qr50=ArchGDAL.getfield(feature, 25)
            # qr20=ArchGDAL.getfield(feature, 24)

            # qh20=ArchGDAL.getfield(feature,48)
            # qh50=ArchGDAL.getfield(feature,49)
            qh100=ArchGDAL.getfield(feature,50)

            xx=ArchGDAL.getfield(feature, 3)
            yy=ArchGDAL.getfield(feature, 4)

            isevaq=ArchGDAL.getfield(feature,59)
            accareai=ArchGDAL.getfield(feature,60)
            toci=ArchGDAL.getfield(feature,61)


            isok = !any([ismissing(name), ismissing(dn2rid),ismissing(qr100)])

            if isok
                push!(BasinName,name)
                push!(reachid,dn2rid)
                
                push!(Qg100,qr100)
                # push!(Qg50,qr50)
                # push!(Qg20,qr20)
                push!(Qhc100,qh100)
                # push!(Qhc50,qh50)
                # push!(Qhc20,qh20)
                push!(nztmx,xx)
                push!(nztmy,yy)
                push!(isEVA,isevaq)
                push!(toc,toci)
                push!(accarea,accareai)
            end
        end
    end 

    QTN100=zeros(length(Qg100),length(durations));
    # QTN50=zeros(size(Qg100));
    # QTN20=zeros(size(Qg100));
    # Durqmax=zeros(size(Qg100));

    # Read Henderson and Collins data


    # Read the topnet results only select the durations that produces the highest peak
    for il=1:length(Qg100)
        
        durrC=0.0
        for (id,dur) in enumerate(durations)
            rint=EVAref
            q=zeros(size(EVAref))
            try
            
                rint,q=GetARI(BasinName[il],dur,reachid[il],EVAref=EVAref,backupfolder=backupfolder)
            catch
                @warn "Issue here: $(BasinName[il]),$dur,$(reachid[il])"
            end
            QTN100[il,id]=q[1]
        end
        
    end
    
    # Output to a readable file
    open(outfile,"w") do io
        Printf.@printf(io,"nztmx,nztmy,dn2reachid,isEVA,Basin,AccArea,TOC,Q100FS,Q100HC")
        for (id,dur) in enumerate(durations)
            Printf.@printf(io,",QTN100:%d",dur)
        end
        Printf.@printf(io,"\n")






        for il=1:length(Qg100)
            Printf.@printf(io,"%f,%f,%d,%d,%s,%f,%f,%f,%f",nztmx[il],nztmy[il],reachid[il],isEVA[il],BasinName[il],accarea[il],toc[il],Qg100[il],Qhc100[il]);
            for (id,dur) in enumerate(durations)
                Printf.@printf(io,",%f",QTN100[il,id])
            end
            Printf.@printf(io,"\n")
   
        end
   
    end
   

end
function GetFloodInfo(Floodfile)
    #
    dataset = ArchGDAL.read(Floodfile);
    layer=ArchGDAL.getlayer(dataset,0)

    nfeat = ArchGDAL.nfeature(layer);

    BasinName=Vector{String}(undef,0)
    reachid=Vector{Int64}(undef,0)

    nztmx=Vector{Int64}(undef,0)
    nztmy=Vector{Int64}(undef,0)
    isEVA=Vector{Bool}(undef,0)
    toc=Vector{Float64}(undef,0)
    accarea=Vector{Float64}(undef,0)

    for i=1:(nfeat)
		
        ArchGDAL.getfeature(layer, i) do feature
            name = ArchGDAL.getfield(feature, 39) # WARNING HARD WIRED
            dn2rid = ArchGDAL.getfield(feature, 42) # WARNING HARD WIRED
            
            qr100=ArchGDAL.getfield(feature, 26)
            # qr50=ArchGDAL.getfield(feature, 25)
            # qr20=ArchGDAL.getfield(feature, 24)

            # qh20=ArchGDAL.getfield(feature,48)
            # qh50=ArchGDAL.getfield(feature,49)
            qh100=ArchGDAL.getfield(feature,50)

            xx=ArchGDAL.getfield(feature, 3)
            yy=ArchGDAL.getfield(feature, 4)

            isevaq=ArchGDAL.getfield(feature,59)
            accareai=ArchGDAL.getfield(feature,60)
            toci=ArchGDAL.getfield(feature,61)


            isok = !any([ismissing(name), ismissing(dn2rid),ismissing(qr100)])

            if isok
                push!(BasinName,name)
                push!(reachid,dn2rid)
                
                # push!(Qg100,qr100)
                # # push!(Qg50,qr50)
                # # push!(Qg20,qr20)
                # push!(Qhc100,qh100)
                # # push!(Qhc50,qh50)
                # push!(Qhc20,qh20)
                push!(nztmx,xx)
                push!(nztmy,yy)
                push!(isEVA,isevaq)
                push!(toc,toci)
                push!(accarea,accareai)
            end
        end
    end 

    return BasinName,reachid,nztmx,nztmy,isEVA,toc,accarea
end

function GetrainfallforQloc(Floodfile;qfile="streamq",backupfolder="/nesi/nobackup/niwa03440/Cylc-Workflow-Outputs/Production_Run2",EVAref=[100],durations=[6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72],outroot="FlowCheck_Rain")

    #Read the geopackage file
    #names,EvDur,EVARID,QEVA=GetDurations(gpkgfile, durations=durations,getgauge=getgauge)

    
    # Read the Flood stats data
    
    dataset = ArchGDAL.read(Floodfile);
    layer=ArchGDAL.getlayer(dataset,0)

    nfeat = ArchGDAL.nfeature(layer);

    BasinName=Vector{String}(undef,0)
    reachid=Vector{Int64}(undef,0)
    
    # Qg100=Vector{Float64}(undef,0)
    # Qg50=Vector{Float64}(undef,0)
    # Qg20=Vector{Float64}(undef,0)
    # Qhc100=Vector{Float64}(undef,0)
    # Qhc50=Vector{Float64}(undef,0)
    # Qhc20=Vector{Float64}(undef,0)
    nztmx=Vector{Int64}(undef,0)
    nztmy=Vector{Int64}(undef,0)
    isEVA=Vector{Bool}(undef,0)
    toc=Vector{Float64}(undef,0)
    accarea=Vector{Float64}(undef,0)

    for i=1:(nfeat)
		
        ArchGDAL.getfeature(layer, i) do feature
            name = ArchGDAL.getfield(feature, 39) # WARNING HARD WIRED
            dn2rid = ArchGDAL.getfield(feature, 42) # WARNING HARD WIRED
            
            qr100=ArchGDAL.getfield(feature, 26)
            # qr50=ArchGDAL.getfield(feature, 25)
            # qr20=ArchGDAL.getfield(feature, 24)

            # qh20=ArchGDAL.getfield(feature,48)
            # qh50=ArchGDAL.getfield(feature,49)
            qh100=ArchGDAL.getfield(feature,50)

            xx=ArchGDAL.getfield(feature, 3)
            yy=ArchGDAL.getfield(feature, 4)

            isevaq=ArchGDAL.getfield(feature,59)
            accareai=ArchGDAL.getfield(feature,60)
            toci=ArchGDAL.getfield(feature,61)


            isok = !any([ismissing(name), ismissing(dn2rid),ismissing(qr100)])

            if isok
                push!(BasinName,name)
                push!(reachid,dn2rid)
                
                # push!(Qg100,qr100)
                # # push!(Qg50,qr50)
                # # push!(Qg20,qr20)
                # push!(Qhc100,qh100)
                # # push!(Qhc50,qh50)
                # push!(Qhc20,qh20)
                push!(nztmx,xx)
                push!(nztmy,yy)
                push!(isEVA,isevaq)
                push!(toc,toci)
                push!(accarea,accareai)
            end
        end
    end 



    
    # QTN50=zeros(size(Qg100));
    # QTN20=zeros(size(Qg100));
    # Durqmax=zeros(size(Qg100));

    # Read Henderson and Collins data


    # Read the topnet results only select the durations that produces the highest peak
    for il=1:length(reachid)
    try
        time,q=GetTNTS(BasinName[il],durations[1],reachid[il],EVAref[1], backupfolder=backupfolder,qfile="streamq")
        open(outroot*"_Flow_"*BasinName[il]*"_"*string(reachid[il])*".txt","w") do io
            Printf.@printf(io,"# Flow for each duration (one per line in this order): \t",);
            for Dur in durations
                Printf.@printf(io," %d\t",Dur);
            end
            Printf.@printf(io,"# Time: Hourly\t");
            
            for ddii=1:length(time)
                #Printf.@printf(io," %f\t",time[ddii]);
    
            end
            Printf.@printf(io,"\n")
    
        end

        open(outroot*"_Rain_"*BasinName[il]*"_"*string(reachid[il])*".txt","w") do io
            Printf.@printf(io,"# Rain for each duration (one per line in this order): \t",);
            for Dur in durations
                Printf.@printf(io," %d\t",Dur);
            end
            Printf.@printf(io,"# Time: Hourly\t");
            
            for ddii=1:length(time)
                #Printf.@printf(io," %f\t",time[ddii]);
    
            end
            Printf.@printf(io,"\n")
    
        end
    
        

        for Dur in durations
        
            time,q=GetTNTS(BasinName[il],Dur,reachid[il],EVAref[1], backupfolder=backupfolder,qfile="streamq")

            timer,rain=GetTNTS(BasinName[il],Dur,reachid[il],EVAref[1], backupfolder=backupfolder,qfile="tseries",varname="aprecip",israin=true)



            open(outroot*"_Flow_"*BasinName[il]*"_"*string(reachid[il])*".txt","a") do io
                for ddii=1:length(time)
                    Printf.@printf(io,"%f\t",q[ddii]);
        
                end
                Printf.@printf(io,"\n")
        
            end

            open(outroot*"_Rain_"*BasinName[il]*"_"*string(reachid[il])*".txt","a") do io
                for ddii=1:length(timer)
                    Printf.@printf(io,"%f\t",rain[ddii]);
        
                end
                Printf.@printf(io,"\n")
        
            end


        end

    catch
        @warn "Issue reading TopNet file for "*BasinName[il]*"_"*string(reachid[il])
    end

        
    end
    
    # Output to a readable file
    
   

end

