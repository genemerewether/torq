#!/usr/bin/awk -f

BEGIN { FS = "," }

{ out="- t: "$NF"\n  x: [" }

{ for(i = 1; i <= NF-2; i++) { out=out""$i", " } }
{ out=out""$(NF-1) }

{ print out"]" }
