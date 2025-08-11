module ut_reconstr
import Dates, DelimitedFiles, NetCDF, Statistics
export coef_s,opt_s,makelind,ut_reconstr1,Predicttidetime,Readharmonicsnc,exceedencecurve

    struct shallow_s
        iconst
        coef
        iname
    end

    struct const_s
        name
        freq
        kmpr
        ikmpr
        df
        doodson
        semi
        isat
        nsat
        ishallow
        nshallow
        doodsonamp
        doodsonspecies
    end

    struct sat_s
        deldood
        phcorr
        amprat
        ilatfac
        iconst
    end

    # first build a constituent object
    struct coef_s
        #constituent parameter
        Name
        A
        A_ci
        g
        g_ci
        freq
        lind
        mean
        slope
        lat
        reftime

    end

    struct opt_s
        constfolder::String
        twodim::Bool
        minsnr::Float64
        minpe::Float64
        nodsatlint::Bool# 0
        nodsatnone::Bool# 0
        gwchlint::Bool# 0
        gwchnone::Bool# 0
        notrend::Bool
    end

    function load_Constants(folder)
        #constt,sat,shallow=load_Constants()

        #folder="D:\\Misc\\Julia\\";
        shallowdata=DelimitedFiles.readdlm(folder * "_ut_shallow.txt";skipstart=1);
        shallow=shallow_s(shallowdata[:,1],shallowdata[:,2],shallowdata[:,3]);

        satdata=DelimitedFiles.readdlm(folder * "_ut_sat.txt";skipstart=1);
        sat=sat_s(satdata[:,1:3],satdata[:,4],satdata[:,5],satdata[:,6],satdata[:,7]);

        constdata=DelimitedFiles.readdlm(folder * "_ut_const.txt";skipstart=1);

        constt=const_s(convert.(String,constdata[:,1]),convert.(Float64,constdata[:,2]),convert.(String,constdata[:,3]),convert.(Float64,constdata[:,4]),convert.(Float64,constdata[:,5]),convert.(Float64,constdata[:,6:11]),convert.(Float64,constdata[:,12]),convert.(Float64,constdata[:,13]),convert.(Float64,constdata[:,14]),convert.(Float64,constdata[:,15]),convert.(Float64,constdata[:,16]),convert.(Float64,constdata[:,17]),convert.(Float64,constdata[:,18]));



        return constt,sat,shallow


    end
    function  ut_astron(jd)
        # % UT_ASTRON()
        # % calculate astronomical constants
        # % input
        # %   jd = time [datenum UTC] (1 x nt)
        # % outputs
        # %   astro = matrix [tau s h p np pp]T, units are [cycles] (6 x nt)
        # %   ader = matrix of derivatives of astro [cycles/day] (6 x nt)
        # % UTide v1p0 9/2011 d.codiga@gso.uri.edu
        # % (copy of t_astron.m from t_tide, Pawlowicz et al 2002)

        nt=length(jd);
        d=Dates.datetime2julian.(jd).-Dates.datetime2julian(Dates.DateTime(1899,12,31,12,0,0));
        D=d./10000;

        args=[ones(nt) d D.*D D.^3]
        sc= [ 270.434164,13.1763965268,-0.0000850, 0.000000039];
        hc= [ 279.696678, 0.9856473354, 0.00002267,0.000000000];
        pc= [ 334.329556, 0.1114040803,-0.0007739,-0.00000026];
        npc=[-259.183275, 0.0529539222,-0.0001557,-0.000000050];
        ppc=[ 281.220844, 0.0000470684, 0.0000339, 0.000000070];

        astro=zeros(6,nt);
        ader=zeros(6,nt);

        #astro=rem( [sc,hc,pc,npc,ppc].*args./360.0 ,1);
        astro[2:6,:]=rem.( [sc hc pc npc ppc]'*args'./360.0 ,1);



        tau=rem.(Dates.datetime2unix.(jd)./(3600*24),1)+astro[3,:]-astro[2,:];
        astro[1,:]=tau;
        dargs=[zeros(nt) ones(nt) 2.0e-4.*D 3.0e-4.*D.*D];
        ader[2:6,:]=[sc hc pc npc ppc]'*dargs'./360.0;
        dtau=1.0.+ader[3,:]-ader[2,:];
        ader[1,:]=dtau;

        return astro,ader
    end

    function ut_FUV(t,tref,lind,lat,ngflgs,folder)
        # % UT_FUV()
        # % compute nodal/satellite correction factors and astronomical argument
        # % inputs
        # %   t = times [datenum UTC] (nt x 1)
        # %   tref = reference time [datenum UTC] (1 x 1)
        # %   lind = list indices of constituents in ut_constants.mat (nc x 1)
        # %   lat = latitude [deg N] (1 x 1)
        # %   ngflgs = [NodsatLint NodsatNone GwchLint GwchNone] each 0/1
        # % output
        # %   F = real nodsat correction to amplitude [unitless] (nt x nc)
        # %   U = nodsat correction to phase [cycles] (nt x nc)
        # %   V = astronomical argument [cycles] (nt x nc)
        # % UTide v1p0 9/2011 d.codiga@gso.uri.edu
        # % (uses parts of t_vuf.m from t_tide, Pawlowicz et al 2002)

        nt = length(t);
        nc = length(lind);
        # nodsat
        if ngflgs[2] # none
            # F = ones(nt,nc);
            # U = zeros(nt,nc);
        else
            if ngflgs[1] # linearized times
                #tt = tref;
            else         # exact times
                tt = t;
            end
            ntt = length(tt);
            constt,sat,shallow=load_Constants(folder);
            astro,dummy=ut_astron(tt);
            if abs(lat)<5
                lat=sign(lat).*5;
            end
            slat=sin(pi*lat/180);
            rr=sat.amprat;
            j=sat.ilatfac.==1;
            rr[j]=rr[j].*0.36309.*(1.0-5.0.*slat.*slat)./slat;
            j=(sat.ilatfac.==2);
            rr[j]=rr[j].*2.59808.*slat;
            uu=rem.( sat.deldood*astro[4:6,:]+sat.phcorr*ones(1,ntt), 1);
            nfreq=length(constt.isat); #ok
            mat = rr*ones(1,ntt).*exp.(1im*2*pi*uu);
            F = ones(ComplexF64,nfreq,ntt);
            ind = unique(sat.iconst);
            for i = 1:length(ind)
                F[Int64(ind[i]),:] = 1 .+sum(mat[sat.iconst.==ind[i],:],dims=1);
            end
            U = imag(log.(F))/(2*pi); # faster than angle(F)
            F=abs.(F);

            for k in findall(isfinite.(constt.ishallow))
                ik=convert.(Int64,constt.ishallow[k].+collect(0:(constt.nshallow[k]-1)));
                j = convert.(Int64,shallow.iname[ik]);
                exp1 = shallow.coef[ik];
                exp2 = abs.(exp1);
                F[k,:]=prod(F[j,:].^(exp2*ones(ntt,1)'),dims=1);
                U[k,:]=sum(U[j,:].*(exp1*ones(ntt,1)'),dims=1);
            end
            F=F[lind,:];
            U=U[lind,:];
            if ngflgs[1] # nodal/satellite with linearized times
                # F = F(ones(nt,1),:);
                # U = U(ones(nt,1),:);
            end
        end
        # gwch (astron arg)
        if ngflgs[4] # none (raw phase lags not greenwich phase lags)
            # if ~exist('const','var')
            #     load('ut_constants.mat','const');
            # end
            # [~,ader] = ut_astron(tref);
            # ii=isfinite(const.ishallow);
            # const.freq(~ii) = (const.doodson(~ii,:)*ader)/(24);
            # for k=find(ii)'
            #     ik=const.ishallow(k)+(0:const.nshallow(k)-1);
            #     const.freq(k)=sum(const.freq(shallow.iname(ik)).*shallow.coef(ik));
            # end
            # V = 24*(t-tref)*const.freq(lind)';
        else
            if ngflgs[3]  # linearized times
                # tt = tref;
            else
                tt = t;   # exact times
            end
            ntt = length(tt);
            # if exist('astro','var')
            #     if ~isequal(size(astro,2),ntt)
            #         [astro,~]=ut_astron(tt');
            #     end
            # else
            #     [astro,~]=ut_astron(tt');
            # end
            # if ~exist('const','var')
            #     load('ut_constants.mat');
            # end
            V=rem.( constt.doodson*astro+constt.semi*ones(1,ntt), 1);
            for k in findall(isfinite.(constt.ishallow))

                ik = convert.(Int64,constt.ishallow[k].+collect(0:(constt.nshallow[k]-1)));
                j = convert.(Int64,shallow.iname[ik]);
                exp1 = shallow.coef[ik];
                V[k,:] = sum(V[j,:].*(exp1*ones(ntt,1)'),dims=1);#sum(V[j,:].*exp1*ones(ntt,1)),1);
            end
            V=V[lind,:];
            if ngflgs[3]    # linearized times
                # [~,ader] = ut_astron(tref);
                # ii=isfinite(const.ishallow);
                # const.freq(~ii) = (const.doodson(~ii,:)*ader)/(24);
                # for k=find(ii)'
                #     ik=const.ishallow(k)+(0:const.nshallow(k)-1);
                #     const.freq(k)=sum( const.freq(shallow.iname(ik)).* ...
                #         shallow.coef(ik) );
                # end
                # V = V(ones(1,nt),:) + 24*(t-tref)*const.freq(lind)';
            end
        end
        return F,U,V
    end

    function ut_E(t,tref,frq,lind,lat,ngflgs,prefilt,folder)
        # UT_E()
        # compute complex exponential basis function
        # inputs
        #   t = times [datenum UTC] (nt x 1)
        #   tref = reference time [datenum UTC] (1 x 1)
        #   frq = frequencies [cph] (nc x 1)
        #   lind = list indices of constituents in ut_constants.mat (nc x 1)
        #   lat = latitude [deg N] (1 x 1)
        #   ngflgs = [NodsatLint NodsatNone GwchLint GwchNone] each 0/1
        #       ([0 1 0 1] case not allowed, and not needed, in ut_E)
        #   prefilt = 'prefilt' input to ut_solv
        # output
        #   E = complex exponential basis function [unitless] (nt x nc)
        # UTide v1p0 9/2011 d.codiga@gso.uri.edu

        nt = length(t);
        nc = length(lind);
        if ngflgs[2] && ngflgs[4]
            F = ones(nt,nc);
            U = zeros(nt,nc);
            V = 24*(t-tref)*frq';
        else
            F,U,V = ut_FUV(t,tref,lind,lat,ngflgs,folder);
        end
        E = F.*exp.(1im*(U+V)*2*pi);
        if !isempty(prefilt)
            # P=interp1(prefilt.frq,prefilt.P,frq)';
            # P( P>max(prefilt.rng) | P<min(prefilt.rng) | isnan(P) )=1;
            # E = E.*P(ones(nt,1),:);
        end
        return E
    end




    function ut_reconstr1(t,coef,opt)
        # single tide reconstruction based on UT
        # UT_RECONSTR1()
        # Reconstruction for a single record. See comments for UT_RECONSTR().
        # UTide v1p0 9/2011 d.codiga@gso.uri.edu

        #fprintf('ut_reconstr: ');

        # parse inputs and options
        #[t,opt] = ut_rcninit(tin,varargin);

        #Preallocate a few vars




        # determine constituents to include
        if !isempty([])
            # [~,ind] = ismember(cellstr(opt.cnstit),coef.name);
            # if ~isequal(length(ind),length(cellstr(opt.cnstit)))
            #     error(['ut_reconstr: one or more of input constituents Cnstit '...
            #         'not found in coef.name']);
            # end
        else
            SNR=zeros(length(coef.A));
            PE=zeros(length(coef.A));
            #ind = collect(1:length(allcon));
            if opt.twodim
                # SNR = (coef.Lsmaj.^2 +coef.Lsmin.^2)./((coef.Lsmaj_ci/1.96).^2 + (coef.Lsmin_ci/1.96).^2);
                # PE = sum(coef.Lsmaj.^2 + coef.Lsmin.^2);
                # PE = 100*(coef.Lsmaj.^2 + coef.Lsmin.^2)/PE;
            else

                SNR = (coef.A.^2)./((coef.A_ci/1.96).^2);
                PE = 100*coef.A.^2/sum(coef.A.^2);


            end
            ind = (SNR.>=opt.minsnr) .& (PE.>=opt.minpe);
        end



        #############################
        # complex coefficients
        rpd = pi/180;
        if opt.twodim
            # ap = 0.5*(coef.Lsmaj(ind) + coef.Lsmin(ind)) .* exp(1i*(coef.theta(ind) - coef.g(ind))*rpd);
            # am = 0.5*(coef.Lsmaj(ind) - coef.Lsmin(ind)) .* exp(1i*(coef.theta(ind) + coef.g(ind))*rpd);
        else
            #ap=zeros(ComplexF64,length(A))


            ap = 0.5*coef.A[ind].*exp.(-1im .*coef.g[ind].*rpd);
            am = conj.(ap);

        end




        # exponentials
        ngflgs = [opt.nodsatlint opt.nodsatnone opt.gwchlint opt.gwchnone];
        #fprintf('prep/calcs ... ');

        prefilt=[];

        #F,U,V = ut_FUV(t,coef.reftime,coef.lind,coef.lat,ngflgs)

        E = ut_E(t,coef.reftime,coef.freq[ind],coef.lind[ind],coef.lat,ngflgs,prefilt,opt.constfolder);
        #E = ut_E(t,coef.reftime,coef.freq,coef.lind,coef.lat,ngflgs,prefilt);

        # fit
        #


        fit = permutedims(E)*ap + conj(transpose(E))*am;

        # mean (& trend)
        u = NaN*ones(size(t));
        whr = .!isnan.(Dates.datetime2unix.(t)); # Absurd statement
        if opt.twodim
            # v = u;
            # if coef.aux.opt.notrend
            #     u(whr) = real(fit) + coef.umean;
            #     v(whr) = imag(fit) + coef.vmean;
            # else
            #     u(whr) = real(fit) + coef.umean + ...
            #         coef.uslope*(t-coef.aux.reftime);
            #     v(whr) = imag(fit) + coef.vmean + ...
            #         coef.vslope*(t-coef.aux.reftime);
            # end
        else
             if opt.notrend
                u[whr] = real(fit) .+ coef.mean;
             else
                u[whr] = real(fit) .+ coef.mean .+ coef.slope*(Dates.datetime2unix.(t).-Dates.datetime2unix(coef.reftime))./(3600*24);
             end
            # v = [];
        end

        return u
    end


    function findlocalmaxima(signal::Vector)
       inds = Int[]
       if length(signal)>1
           if signal[1]>signal[2]
               push!(inds,1)
           end
           for i=2:length(signal)-1
               if signal[i-1]<signal[i]>signal[i+1]
                   push!(inds,i)
               end
           end
           if signal[end]>signal[end-1]
               push!(inds,length(signal))
           end
       end
       inds
     end

     function findlocalminima(signal::Vector)
         subsign=signal.*-1.0;
         return findlocalmaxima(subsign);
     end
    function Predicttidetime(t,coef,opt)
        #Return high and low tide time and level

        # first predict the tide at 1 minute interval
        # need to do that 1 day before and 1 day after what is requested
        tint=collect(floor(minimum(t)-Dates.Day(1), Dates.Day):Dates.Minute(1):ceil(maximum(t)+Dates.Day(1), Dates.Day));

        #predict tide
        sl=ut_reconstr1(tint,coef,opt);

        #Find local extrema
        HTind=findlocalmaxima(sl);
        LTind=findlocalminima(sl);

        #Use only extrem between the dates initially selected
        if length(t)==1
            #
            index=(tint[HTind].>=floor(t, Dates.Day) .&  tint[HTind].<ceil(t, Dates.Day));
            HTind=HTind[index];
            index=(tint[LTind].>=floor(t, Dates.Day) .&  tint[LTind].<ceil(t, Dates.Day));
            LTind=LDind[index];
        else
            #
            index=(tint[HTind].>=t[1]) .& (tint[HTind].<=t[end]);
            HTind=HTind[index];
            index=(tint[LTind].>=t[1]) .& (tint[LTind].<=t[end]);
            LTind=LTind[index];
        end

        #
        return tint[HTind],sl[HTind],tint[LTind],sl[LTind]

    end

    function makelind(coef,opt)
        #Fill-in lind array based on Coef.name.
        constt,sat,shallow=load_Constants(opt.constfolder);
        newlind=zeros(Int64,length(coef.Name));
        for n=1:length(coef.Name)
            index=findfirst(strip(coef.Name[n]).==constt.name)
            if index==nothing
                newlind[n]=1;
            else
                newlind[n]=index;
            end

        end
        return newlind;

    end
    function Readharmonicsnc(ncfile, constfolder)
        #Read harmonics data and metadata from Netcdf file
        #ncfile="D:\\Projects\\Tonga\\Waterlevel\\NiuatoputapuTideCst-d.nc"
        NamesChar = string.(strip.(NetCDF.nc_char2string(NetCDF.ncread(ncfile,"Name"))));
        # Convert NETCDF ASCII CHAR TO Strings


        A = NetCDF.ncread(ncfile,"A");
        A_ci = NetCDF.ncread(ncfile,"A_ci");
        g = NetCDF.ncread(ncfile,"g");
        g_ci = NetCDF.ncread(ncfile,"g_ci");
        freq = NetCDF.ncread(ncfile,"freq");

        MSL=NetCDF.ncgetatt(ncfile, "Global", "MSL")
        trend=NetCDF.ncgetatt(ncfile, "Global", "trend")
        reftime=Dates.DateTime(NetCDF.ncgetatt(ncfile, "Global", "reftime"))
        Lat=NetCDF.ncgetatt(ncfile, "Global", "Latitude")

        tempcoef=coef_s(NamesChar,A,A_ci,g,g_ci,freq,zeros(length(freq)),MSL,trend,Lat,reftime)
        #opt=opt_s("D:\\Projects\\Tonga\\tonga-ocean-forecasting-tools\\Tide-Calendars\\",false,2.0,0.0,false,false,false,false,false)
        opt=opt_s(constfolder,false,2.0,0.0,false,false,false,false,false)

        linds=makelind(tempcoef,opt);

        return coef_s(NamesChar,A,A_ci,g,g_ci,freq,linds,MSL,trend,Lat,reftime)



    end

    function exceedencecurve(coef,opt;nyear=18)
        # Calculate 100-year tidal exceedence curve for High tide and low tide based on tide constituents
        # return 2 vector of tidal exceedance for every percentile with 0:1:100
        # WARNING this analysis may take a while


        # HTLTall = Float64[]
        #
        #
        HTall = Float64[];
        LTall = Float64[];

        HTexcurve=zeros(101);
        LTexcurve=zeros(101);

        for year=2000:(2000+nyear)
             t=[Dates.DateTime(year,01,01,0,0,0) Dates.DateTime(year,12,31,23,59,59)];

             HT,Hlev,LT,Llev=Predicttidetime(t,coef,opt);
             for a in Hlev
                 push!(HTall,a)
             end
             for a in Llev
                 push!(LTall,a)
             end

        end

        #
        HTexcurve[101]=maximum(HTall);
        HTexcurve[1]=minimum(HTall);
        for n=1:99
            HTexcurve[n+1]=Statistics.quantile(HTall,n/100);
        end
        LTexcurve[101]=maximum(LTall);
        LTexcurve[1]=minimum(LTall);
        for n=1:99
            LTexcurve[n+1]=Statistics.quantile(LTall,n/100);
        end

        return HTexcurve,LTexcurve;
    end




end




#Example
#
# opt=opt_s("C:\\Users\\bosserellec\\Documents\\GitHub\\Julia_mixbag\\",false,2.0,0.0,false,false,false,false,false)
#
#
# Names=["M2" "N2" "S2" "K1" "O1" "NU2" "MU2" "P1" "2N2" "K2" "L2" "Q1" "M3" "MKS2" "EPS2" "LDA2" "J1" "SK3" "MS4" "NO1" "M4" "OQ2" "2MS6" "OO1" "M6" "THE1" "SO1" "PHI1" "UPS1" "SIG1" "TAU1" "SO3" "BET1" "MN4" "2SK5" "2Q1" "MK3" "RHO1" "2MN6" "MK4" "ETA2" "MSN2" "2MK6" "S4" "2SM6" "MO3" "SSA" "ALP1" "SK4" "SN4" "2MK5" "MSF" "MM" "MSM" "CHI1" "MSK6" "3MK7" "MF" "M8"]
#
# A=vec([0.493138384657320 0.122220407530187 0.0625455828349263 0.0624069393945612 0.0319278720784309 0.0264898452077264 0.0198967110349878 0.0193689717281431 0.0168389355226699 0.0163837418565904 0.00505401616918255 0.00480818731027601 0.00475331790152582 0.00462103544809817 0.00418090823059413 0.00352786018482124 0.00328474197224013 0.00299819494350093 0.00267398578239277 0.00222404439626272 0.00221481509990188 0.00202827962980981 0.00200951693684904 0.00169121773465207 0.00168236883535281 0.00159827419045768 0.00142463237217343 0.00109577059578159 0.00107846119006118 0.000993189286515915 0.000975633183598672 0.000918336597853578 0.000894227206344413 0.000867043932378154 0.000822231068028090 0.000800455613275940 0.000775203721978753 0.000732787870909778 0.000730892720022018 0.000642278338886823 0.000641631500624715 0.000592223723091890 0.000584837923589339 0.000511274861454928 0.000433699506638236 0.000407137846795978 0.000349049337361200 0.000346090587423473 0.000315558376262097 0.000298038267674244 0.000278980485781384 0.000249667477715449 0.000217132434292240 0.000180407802542512 0.000167720213574086 0.000119570299593473 0.000105539763535811 6.97852998023546e-05 3.44081585416120e-05])
#
#
# A_ci=vec([0.000588412284822346 0.000449550463417388 0.000578787340178285 0.00102801034135234 0.000901721376152978 0.000585423188152217 0.000483878821705129 0.00100428741461252 0.000536316279001065 0.000528567136260811 0.000557159648900962 0.000983518880624109 0.000380423461000557 0.000483825413427917 0.000615262454783609 0.000519315316376166 0.00102152209349214 0.000378377519189604 0.000277177036685191 0.000892607205617728 0.000290023838393219 0.000577605848346834 0.000157787122256716 0.00103672984663083 0.000166869447236043 0.00108130453248096 0.000979178365422599 0.000992046886662798 0.000812350033397252 0.000870678021316109 0.000907997233172556 0.000369550758945878 0.000867683288921515 0.000271926645768515 0.000163300640777281 0.000923975718625942 0.000398768279644157 0.000888418911306774 0.000173031439720871 0.000265467282237147 0.000606188010231063 0.000580009876029937 0.000163953246041184 0.000322027299520705 0.000201901560421842 0.000345312880500407 0.000172792737870738 0.000622538958918522 0.000252605584824177 0.000264108816971921 0.000168209986957173 0.000146719184967278 0.000160264048046739 0.000146318881279309 0.000574211721978646 0.000139940142057110 8.36494108371187e-05 0.000115817993922310 5.25293082784274e-05]);
#
#
# g=vec([206.472841113229 189.592580251570 228.577462497562 226.373348002011 234.835289824733 187.839560476596 168.304522584086 225.557939454922 178.394293158333 205.897114497219 215.311727831223 234.076201189012 229.206126637714 116.279828605487 166.928712508262 246.030955498636 214.551246561116 260.314054030127 310.338504114343 214.490110505745 275.280689204499 176.380397128870 9.20242692250800 255.683532644545 6.56915278799839 233.464930099497 180.491345742988 129.971659759884 112.443436991686 207.206716473664 62.0453964001101 338.538332765595 258.338812986228 254.295030014493 161.959834408184 173.887665321566 308.049732087583 158.438470723781 350.347560816338 256.377800075036 135.213419276068 213.090201785295 4.50300359535021 18.3477087351634 40.5006471268699 199.143608706706 157.887516216598 69.1261949804776 102.327941700325 317.739078406793 152.838849373427 314.204463395353 200.332540669770 170.900964010534 170.603322266183 273.618085775180 20.1614848496786 322.451311705972 35.7582443751978]);
#
#
# g_ci=vec([0.0651156148667687 0.248441538866563 0.485431942020961 0.871333797973204 1.62594515196797 1.20033782012330 1.40641743739766 2.76362215187178 1.57291814434557 1.88803924485863 5.28229777797954 12.4050990000139 4.82313481933750 6.62187807229875 7.96130144857590 8.18982885983305 17.3288572334094 7.16701207670834 5.32836753987433 26.4180730227022 7.52256768787866 15.8598106693538 5.16590633645002 30.3883739218752 5.47195767090829 29.8346432117955 39.8780336598458 53.6505051492318 55.4988652945298 52.6174627309777 66.3909486694234 24.0142762976939 58.4211411888950 18.4233983898710 13.6239107264190 65.9751036006228 30.5017797719044 86.4888725780996 12.5038974085096 24.4683746049129 48.5053867117686 62.0396418912288 24.2869938969906 30.7799282282531 24.8819491867134 45.8205210075085 22.9503081356430 152.927725429452 54.4011619209159 58.4178036573127 31.8309862319390 29.8719972296119 43.3996265495481 49.2415160697615 223.046195968590 84.2252019397368 49.7433247442838 132.879682332586 138.940203905053]);
#
# freq=vec([0.0805114006717706 0.0789992486986775 0.0833333333333333 0.0417807462216577 0.0387306544501129 0.0792016199833009 0.0776894680102079 0.0415525871116757 0.0774870967255845 0.0835614924433154 0.0820235526448637 0.0372185024770199 0.120767101007656 0.0807395597817526 0.0761773160371148 0.0818211813602403 0.0432928981947507 0.125114079554991 0.163844734005104 0.0402685942485646 0.161022801343541 0.0759749447524915 0.244356134676875 0.0448308379932025 0.241534202015312 0.0430905269101274 0.0446026788832204 0.0420089053316397 0.0463429899662955 0.0359087217885502 0.0389588135600949 0.122063987783446 0.0400404351385826 0.159510649370448 0.208447412888324 0.0357063505039268 0.122292146893428 0.0374208737616432 0.240022050042219 0.164072893115086 0.0850736444164084 0.0848454853064264 0.244584293786857 0.166666666666667 0.247178067338437 0.119242055121884 0.000228159109982028 0.0343965698154571 0.166894825776649 0.162332582032011 0.202803547565199 0.00282193266156274 0.00151215197309305 0.00130978068846969 0.0404709655331880 0.247406226448419 0.283314948236969 0.00305009177154477 0.322045602687082]);
#
# lind=vec([48 42 57 21 13 43 40 19 39 59 54 11 69 50 35 53 25 74 86 16 82 34 110 28 106 24 27 23 29 10 14 71 15 79 99 9 72 12 103 87 61 60 111 89 113 68 3 8 90 84 96 6 5 4 17 114 120 7 125]);
#
#
# coef=coef_s(Names,A,A_ci,g,g_ci,freq,lind,4.43338652302532e-05,4.10258541806695e-06,-15,DateTime(2019,02,4,7,34,02))
#
# t=collect(DateTime(2009,12,31,11,0,0):Dates.Hour(1):DateTime(2009,12,31,12,0,0));
#
# sl=ut_reconstr1(t,coef,opt);
