#using Pkg

#Pkg.add("NetCDF")


using Dates,Printf,NetCDF,CFTime

include("./ut_reconstr.jl")

using .ut_reconstr


function ftoi(x)
	if x>=0.0
		intcast=trunc(Int, x+0.5)
	else
		intcast=trunc(Int, x-0.5)
	end
	return intcast
end

"""
get_constituents(lon,lat,tidemodel,opt)



read original nz tide constituents
"""
function get_constituents(lon,lat,tidemodel,opt;reflat=-35.0,islatlon=true)
	println("Load lat lon")
	latlonmodel=ncread(tidemodel,"parameters",start=[2,1],count=[2,-1]);
	nodeidx=argmin(hypot.(latlonmodel[1,:].-lat,latlonmodel[2,:].-lon))'
	
	lat4tide=reflat
	if islatlon
		lat4tide=lat;
	end
		
		
	nzmodelcoef=get_constituents(nodeidx,tidemodel,opt,reflat=lat4tide)

	return nzmodelcoef
end

function get_constituents(index,tidemodel,opt;reflat=-35.0)
	#println("Load constituents")
	constituents=ncread(tidemodel,"parameters",start=[1,index],count=[-1,1]);

	cstnames=["M2", "S2", "N2", "K1", "O1", "Q1", "L2", "P1", "MU2", "T2", "K2", "NU2", "2N2"];
	cstfreq=[0.08051140067176844, 0.08333333333333333, 0.07899924869868231, 0.04178074622165745, 0.038730654450111, 0.03721850247702487, 0.08202355264485457, 0.04155258711167587, 0.07768946801020357, 0.08321925922953312, 0.0835614924433149, 0.0792016199832897, 0.0774870967255962];
	cstA=constituents[5:17];
	cstg=constituents[18:(end-1)];
	emptyval=zeros(length(cstfreq))
	tempcoef=coef_s(cstnames,cstA,emptyval,cstg,emptyval,cstfreq,emptyval,0.0,0.0,reflat,DateTime(2000,1,1))

	linds=makelind(tempcoef,opt);

	nzmodelcoef=coef_s(cstnames,cstA,emptyval,cstg,emptyval,cstfreq,linds,0.0,0.0,reflat,DateTime(2000,1,1))

	return nzmodelcoef
end

function bulk_get_constituents(lonstart,lonend,latstart,latend,tidemodel,opt)
	println("Load lat lon")
	
	latlonmodel=ncread(tidemodel,"parameters",start=[2,1],count=[2,-1]);

	#nodeidx=argmin(hypot.(latlonmodel[1,:].-lat,latlonmodel[2,:].-lon))'

	nodeidx=findall((latlonmodel[1,:] .>= latstart) .& (latlonmodel[1,:] .<= latend) .& (latlonmodel[2,:] .>= lonstart) .& (latlonmodel[2,:].<= lonend));

	nodesoidx=sort(nodeidx)
	# check sequentiality to limt calls to NetCDF
	
	# seqidx=Vector{Tuple{Int64,Int64}}();

	# stidx=1
	# ndidx=1
	# for i=1:length(nodeidx)
	# 	if (i==1) .| (nodesoidx[i-1] < nodesoidx[i]-1)
	# 		stidx=i;
	# 	end
	# 	if (i==length(nodeidx)) .| (nodesoidx[i+1] > nodesoidx[i]+1)
	# 		ndidx=i-stidx+1;
	# 		push!(seqidx,tuple(stidx,ndidx))
	# 	end
	# end


	println("Load constituents")

	nzmodelcoef=Vector{coef_s}()

	for si=1:nodeidx
		

		push!(nzmodelcoef,get_constituents(nodeidx[si],tidemodel,opt))
	end

	return nzmodelcoef
end

function bulk_get_constituents(region,res,tidemodel,opt;reflat=-35.0,islatlon=true)
	println("Load lat lon")
	
	latlonmodel=ncread(tidemodel,"parameters",start=[2,1],count=[2,-1]);

	nx,ny=Calcnxny(region,res)

	lon,lat=getxy(region,res);

	nzmodelcoef=Vector{coef_s}()

	latmodel=latlonmodel[1,:];
	lonmodel=latlonmodel[2,:];

	inBB=(latmodel .>= region[3]) .&& (latmodel .<= region[4]) .&& (lonmodel .<= region[2]) .&& (lonmodel .>= region[1])
	
	latinb=latmodel[inBB];
	loninb=lonmodel[inBB];

	indinbb=findall(inBB);
	

	for i=1:nx
		for j=1:ny
			nodeidxinb=argmin(hypot.(latinb.-lat[j],loninb.-lon[i]))'
			nodeidx=indinbb[nodeidxinb]

			uselat=islatlon ? lat[j] : reflat
	

			push!(nzmodelcoef,get_constituents(nodeidx,tidemodel,opt,reflat=uselat));
		end
	end
	return nzmodelcoef
end




function tide2BGF(t,sl,outfile)
	tsec=Dates.value.(t.-t[1])./1000.0;


	open(outfile,"w") do io
    	for i=1:length(t)
         Printf.@printf(io,"%f\t%f\n",tsec[i],sl[i]);

    	end

	end
end

function predictNZtides(lon,lat,outfile; datumshift=-0.15, tidemodelfile="/nesi/project/niwa03150/reevegm/nz_tide_cons/tide_surface_20210923_tidal_cons_output.nc", const_folder="/nesi/project/niwa03440/bosserellec/democycl/data/")

	println("Load Constituents")
	println("opt")
	opt=opt_s(const_folder,false,2.0,0.0,false,false,false,false,false);
	println("nzmodelcoef")
	nzmodelcoef=get_constituents(lon,lat,tidemodelfile,opt,islatlon=false)

	timepred=(DateTime(2000,1,1,0,0,0)-Dates.Day(4)):Dates.Minute(30):(DateTime(2000,1,1,0,0,0)+Dates.Day(4));
	predtide=zeros(length(timepred));
	
	if(sum(nzmodelcoef.A)>0.0)

		println("Calculate MHWS10")
		hT,LT=exceedencecurve(nzmodelcoef,opt,nyear=1)

		println("Find similar tide timeseries")

		HTtime,HTwl,LTtime,LTwl=Predicttidetime([DateTime(2000,01,1) DateTime(2001,12,31)],nzmodelcoef,opt)

		HTdiff=(HTwl.-hT[end-1])

		indx=argmin(abs.(HTdiff))

		timepred=(HTtime[indx]-Dates.Day(4)):Dates.Minute(30):(HTtime[indx]+Dates.Day(4))

		predtide=ut_reconstr1(timepred,nzmodelcoef,opt)
	end


	println("Save to file")
	tide2BGF(timepred, predtide.+datumshift, outfile);
end

function predictMaptide(region,res, refX,refY,outfile;reflat=-35.0,datumshift=-0.15, tidemodelfile="/nesi/project/niwa03150/reevegm/nz_tide_cons/tide_surface_20210923_tidal_cons_output.nc", const_folder="/nesi/project/niwa03440/bosserellec/democycl/data/",reftimestr="")
	#get coordinate for the forcing grid


	if(!isempty(reftimestr))
		reftime=DateTime(reftimestr,"y-m-dTH:M:S")

	else
		reftime=DateTime("2000-01-01T00:00:00","y-m-dTH:M:S")
	end

	println("Load ref Constituents")
	println("opt")
	opt=opt_s(const_folder,false,2.0,0.0,false,false,false,false,false);
	println("nzmodelcoef")
	refmodelcoef=get_constituents(refX,refY,tidemodelfile,opt,islatlon=false,reflat=reflat)

	nzmodelcoef=bulk_get_constituents(region,res,tidemodelfile,opt,reflat=reflat,islatlon=false)

	nx,ny=Calcnxny(region,res)
	println("nx="*string(nx)*" ny="*string(ny))

	if length(nzmodelcoef)!=(nx*ny)
		@warn("size does not match. vec(coef): "*string(length(nzmodelcoef))*" nx*ny= "*string(nx*ny))
	end

	timepred=(DateTime(2000,1,1,0,0,0)-Dates.Day(4)):Dates.Minute(30):(DateTime(2000,1,1,0,0,0)+Dates.Day(4));
	predtide=zeros((nx,ny,length(timepred)));
	
	if(sum(refmodelcoef.A)>0.0)

		println("Calculate MHWS10")
		hT,LT=exceedencecurve(refmodelcoef,opt,nyear=1)

		println("Find similar tide timeseries")

		HTtime,HTwl,LTtime,LTwl=Predicttidetime([DateTime(2000,01,1) DateTime(2001,12,31)],refmodelcoef,opt)

		HTdiff=(HTwl.-hT[end-1])

		indx=argmin(abs.(HTdiff))

		timepred=(HTtime[indx]-Dates.Day(4)):Dates.Minute(30):(HTtime[indx]+Dates.Day(4))
	end

	# We need to move the time relative to the time reference impose by the function not just the real time axis
	

	lon,lat=getxy(region,res);

	for i=1:nx
		for j=1:ny
			#nzmodelcoef=get_constituents(lon[i],lat[j],tidemodelfile,opt)
			predtide[i,j,:]=ut_reconstr1(timepred,nzmodelcoef[j+(i-1)*ny],opt);
		end
	end

	if(isempty(reftimestr))
		timerefvec=timepred
		reftime = timepred[1]
	else
		timerefvec=(reftime-Dates.Day(4)):Dates.Minute(30):(reftime+Dates.Day(4))
	end



	tunits="seconds since "*Dates.format(reftime, "yyyy-mm-dd HH:MM:SS")
	write3nc(lon,lat,timerefvec,predtide.+datumshift,outfile,timeunit=tunits)

	
end

function predictMaptideDate(region,res, startdate,enddate,outfile;datumshift=-0.15,reflat=-35.0, tidemodelfile="/nesi/project/niwa03150/reevegm/nz_tide_cons/tide_surface_20210923_tidal_cons_output.nc", const_folder="/nesi/project/niwa03440/bosserellec/democycl/data/")
	#get coordinate for the forcing grid
	println("Load ref Constituents")
	println("opt")
	opt=opt_s(const_folder,false,2.0,0.0,false,false,false,false,false);
	println("nzmodelcoef")
	
	nzmodelcoef=bulk_get_constituents(region,res,tidemodelfile,opt,islatlon=false,reflat=reflat)
	println("coef loaded")
	nx,ny=Calcnxny(region,res)

	timepred=(startdate:Dates.Minute(10):enddate);
	predtide=zeros((nx,ny,length(timepred)));
	
	

	lon,lat=getxy(region,res);

	for i=1:nx
		for j=1:ny
			#nzmodelcoef=get_constituents(lon[i],lat[j],tidemodelfile,opt)
			predtide[i,j,:]=ut_reconstr1(timepred,nzmodelcoef[j+(i-1)*ny],opt);
		end
	end


	write3nc(lon,lat,timepred,predtide.+datumshift,outfile)

	
end

function write3nc(x,y,t,z,ncfile;varnames=["x","y","t","z"],timeunit="hours since 2000-01-01 00:00:00",varoutname="z")
	xdimid=NcDim("x",length(x),values=x[1:end])
	ydimid=NcDim("y",length(y),values=y[1:end])
	
	timatts = Dict("standard_name" => "time","units"    => timeunit,"axis" => "T","calendar" => "proleptic_gregorian");
	tdimid=NcDim("time",length(t),atts=timatts,values=timeencode(t,timeunit))

	#println(length(x))
	#println(length(y))

	#println(length(t))

	#println(size(z))

	#println(timeencode(t,timeunit))
	UWvarid=NcVar(varoutname,[xdimid,ydimid,tdimid],t=Float32,compress=3)
	#VWvarid=NcVar("Vwind",[xdimid,ydimid,tdimid],t=Float32,compress=3)

	ncid=NetCDF.create(ncfile,[UWvarid], mode=NC_NETCDF4)
	#ncid=NetCDF.create(ncfile,[UWvarid,VWvarid], mode=NC_NETCDF4)


	NetCDF.putvar(ncid,varoutname,Float32.(z),start=[1,1,1],count=[-1,-1,-1])
	#NetCDF.putvar(ncid,"Vwind",Float32.(vw),start=[1,1,1],count=[-1,-1,-1])

	NetCDF.sync(ncid)
	NetCDF.close(ncid)


end

function Puttimeatt(infile,timefile;timeunit="seconds",varname="time")
	

	Raintimeref=""

	# read reftimefile
	open(timefile) do f
		tidetimeref=readline(f)
		Raintimeref=readline(f)
		BGtimerefformated=readline(f)
		BGtimestartformated=readline(f)
	end

	timeunitall=timeunit*" since "*Raintimeref
	
	timatts = Dict("standard_name" => "time","units"    => timeunitall,"axis" => "T","calendar" => "proleptic_gregorian");
	ncputatt(infile,varname,timatts);
end


function getxy(region::NTuple{4,AbstractFloat},res::AbstractFloat)
	nx,ny=Calcnxny(region,res);

	x=fill(NaN,(nx));
	y=fill(NaN,(ny));

	for j = 1:ny
		y[j]=region[3]+(j-1)*res;
	end
	for i=1:nx
		x[i]=region[1]+(i-1)*res;
	end

	return x,y
end

function Checkregion(region::NTuple{4,AbstractFloat},res::AbstractFloat)
	xmin=region[1];
	ymin=region[3];
	xmax=xmin + ceil((region[2] - xmin) / res) * res;
	ymax=ymin + ceil((region[4] - ymin) / res) * res;

	regfixed=(xmin,xmax,ymin,ymax)
	return regfixed
end
# function ftoi(a)
# 	return floor(Int,a+0.5)
# end

function Calcnxny(region::NTuple{4,AbstractFloat},res::AbstractFloat)

	xmin=region[1];
	ymin=region[3];
	xmax=region[2];
	ymax=region[4];


	nx=ftoi((xmax - xmin) / res) + 1
	ny=ftoi((ymax - ymin) / res) + 1


	return nx,ny
end

function Getregion(x,y)

	xmin=minimum(x);
	ymin=minimum(y);
	xmax=maximum(x);
	ymax=maximum(y);

	nx=length(x);
	ny=length(y);

	res=(xmax - xmin)/(nx-1);
	regionbeta=(xmin,xmax,ymin,ymax);

	region=Checkregion(regionbeta,res);

	# nx=Int((xmax - xmin) / res) + 1
	# ny=Int((ymax - ymin) / res) + 1

	return region,res
end
#println("long: "*ARGS[1]*";  lat: "*ARGS[2]*";  output: "*ARGS[3])

#predictNZtides(parse(Float64, ARGS[1]),parse(Float64,ARGS[2]),ARGS[3],tidemodelfile=ARGS[4])
parsetuple(s::AbstractString) = occursin(",",s) ? Tuple(parse.(Float64, split(s, ','))) : parse(Float64,s)

