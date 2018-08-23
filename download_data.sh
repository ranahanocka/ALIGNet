#!/usr/bin/env bash

declare -a glinks=("https://drive.google.com/uc?export=download&id=1Vv-Jz1VpI48MOVgK3Hq6ZYrs2NDP-FQ2" "https://drive.google.com/uc?export=download&id=14Pnrp9ahtRbjEehkI-oM8cBMQKERg6GY")

for gURL in "${glinks[@]}"
do
	# match more than 26 word characters
	ggID=$(echo "$gURL" | egrep -o '(\w|-){26,}')
	echo "$ggID"

	ggURL='https://drive.google.com/uc?export=download'

	curl -sc /tmp/gcokie "${ggURL}&id=${ggID}" >/dev/null  
	getcode="$(awk '/_warning_/ {print $NF}' /tmp/gcokie)"

	cmd='curl --insecure -C - -LOJb /tmp/gcokie "${ggURL}&confirm=${getcode}&id=${ggID}"'
	echo -e "Downloading from "$gURL"...\n"
	eval $cmd
done
