#!/bin/bash
##
## Author:
##		Joe Pelmard
## Description:
##		Script extraction timeseries of zs, h, u, v at location point (x,y)
##
##		ncToQTimeseries.sh <infile> <x> <y> <outfile>
##		<infile>	: input file	-- MUST BE GIVEN
##		<x>			: x-coordinate	-- MUST BE GIVEN
##		<y>			: y-coordinate 	-- MUST BE GIVEN
##		<outfile>	: output file	-- MUST BE GIVEN
##
## Example:
##		ncToQTimeseries.sh BGout.sh 15150.0 51515.0 timeseries.txt
##			-> Output: timeseries.txt
##

module load NCO

infile=$1
x=$2
y=$3
outfile=$4

listvar="zs h u v"

# Delete temporary files if already existing
rm .${outfile}.tmp .timeseries_tmp.nc .var.tmp .${outfile}1.tmp 2>/dev/null
###########################################################################

# Check on which refinement level the point is located
for i in {0..10}
do
	Pi="P${i}"
	ncks -v h_$Pi -d xx_$Pi,$x -d yy_$Pi,$y -d time,0 $infile .check_lvl_tmp.nc
	ncrename -h -O -v h_$Pi,z .check_lvl_tmp.nc
	
	val=$(ncdump -v z .check_lvl_tmp.nc | sed "0,/^data:/d; s/[^0-9]*//g; s/,/\\n/g" | grep -v "^$")

	if [ "$val" != "" ]
	then
		rm .check_lvl_tmp.nc
		break
	fi
	rm .check_lvl_tmp.nc
	
done

echo -e "\n\n ### Generation of timeseries at location ($x, $y)";
echo -e "       + Input file : $infile";
echo -e "       + Output file: $outfile";
echo -e "       + level      : $Pi"


# Start extraction of vars
for var in $listvar
do
	echo -e "\n # Extraction of variable $var..."
	
	# Create netcdf file with only $var at location (x,y)
	ncks -v ${var}_$Pi -d xx_$Pi,${x} -d yy_$Pi,${y} $infile .timeseries_tmp.nc
	ncrename -h -O -v ${var}_$Pi,z .timeseries_tmp.nc

	# If var is the first variable of the list (zs), create temp file .${outfile}.tmp in which variables are stored as column with the time column
	if [ "$var" == "zs" ]
	then
		ncdump -v time       .timeseries_tmp.nc | sed "0,/^data:/d; s/time =//g; s/}//g; s/;//g; s/ //g; s/,/\\n/g" | grep -v "^$" > .${outfile}.tmp
	fi
	ncdump -v z .timeseries_tmp.nc | sed "0,/^data:/d; s/z =//g; s/}//g; s/;//g; s/ //g; s/,/\\n/g" | grep -v "^$" > .var.tmp
	
	# Add new variable into column file (must be done in two steps)
	paste -d'\t' .${outfile}.tmp .var.tmp > .${outfile}1.tmp
	cp .${outfile}1.tmp .${outfile}.tmp
	
	rm .timeseries_tmp.nc .var.tmp .${outfile}1.tmp
done

# Add header
sed -i "1 i\#time[s]\\tzs[m]\\th[m]\\tu[m/s]\\tv[m/s]" .${outfile}.tmp
sed -i "1 i\# x=${x}\\ty=${y}\\t# Time series extracted from BG_Flood netCDF output" .${outfile}.tmp

cp .${outfile}.tmp ${outfile}

rm .${outfile}.tmp
