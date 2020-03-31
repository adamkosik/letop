#!/usr/bin/env bash


source $1/bin/activate
cd ../
lestofire=$(pwd)


for d in $lestofire/examples/*/; do
	if [ "$lestofire/examples/heat_exchanger/" = "$d" ]; then
		echo $d
		cd $d
		python3 2D_mesh.py
		gmsh -2 2D_mesh.geo
		python3 heat_exchanger_al.py > $lestofire/test/output.txt
		cd $lestofire/test/
		python3 check_examples_output.py output.txt heat_exchanger
	fi
done