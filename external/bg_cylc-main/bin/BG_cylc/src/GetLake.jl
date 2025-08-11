using ArchGDAL,Printf

parsetuple(s::AbstractString) = occursin(",",s) ? Tuple(parse.(Float64, split(s, ','))) : parse(Float64,s);

function extractLakes(region,lakesgis;infofile="mylakes.txt",outfolder="")
    # asume region is (xmin,xmax,ymin,ymax)
    dataset = ArchGDAL.read(lakesgis)
    layer = ArchGDAL.getlayer(dataset, 1)

    #geomtype=ArchGDAL.WKBPOLYGON

    #spatialref=

    simplepolygon = ArchGDAL.createpolygon([(region[1],region[3]), (region[2],region[3]), (region[2],region[4]), (region[1],region[4]), (region[1],region[3])])
    open(infofile,"w") do io
        # create empty file 
    end

    for n=1:ArchGDAL.nfeature(layer)
        ArchGDAL.getfeature(layer, n) do feature
            geom=ArchGDAL.getgeom(feature, 0)
            #Lakebbox=ArchGDAL.boundingbox(geom))

            geomtype=ArchGDAL.wkbPolygon

            spatialref=ArchGDAL.getspatialref(geom)

            check=ArchGDAL.intersects(geom,simplepolygon)

            name="Lake_"*string(n)
           # name=ArchGDAL.getfield(feature, 0)
            #println(name," ",check)

           

            if check 
                 #println(name)

                 item="Lake_"*name

                 filename=outfolder*"Lake_"*name*".geojson"

                 n=0

                 while(isfile(filename))
                    n=n+1;
                    filename=outfolder*"Lake_"*name*"_"*string(n)*".geojson"
                    item="Lake_"*name*"_"*string(n)
                 end

                 open(infofile,"a") do io
                    Printf.@printf(io,"%s\n",item)
                 end


                ArchGDAL.create(filename,driver=ArchGDAL.getdriver("geojson")) do mynewlake
                     # Define the name of your layer
                    layer_name = "my_layer"
                    # Create the layer
                    
                    #newlayer = ArchGDAL.create(mynewlake, layer_name, geomtype,spatialref)
                    ArchGDAL.createlayer(name = layer_name, dataset = mynewlake, geom = ArchGDAL.wkbPolygon, spatialref = spatialref) do mylayer
                        #println(name)
                        #newfeature = ArchGDAL.newfeature(newlayer)#
                        ArchGDAL.addfeature!(mylayer,feature)
                        #return nothing
                    end

                    return nothing
                    
                end
                 #println(name)
               
            end
        end
    end
end

