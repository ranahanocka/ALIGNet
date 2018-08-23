#!/usr/bin/env bash

# download model from google drive

declare -a glinks=("https://drive.google.com/uc?export=download&id=110tpXbaMy6B1xajMCMZsfqnpUADn0Cmd" "https://drive.google.com/open?id=1nLa2BUZ6vBKwGDfCgQYrRt2boKuVzqVc")

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

mv vase.t7 vase/weights.t7
mv vase_meanstdCache.t7 vase/meanstdCache.t7