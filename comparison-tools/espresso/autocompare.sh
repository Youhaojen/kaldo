#!/bin/bash
# Assumes modded harmonic_with_q.py file is installed in your kALDo build
# directories
# home=$( realpath . )
md_mods="modded_espresso"
k_mods="modded_kaldo"

# scripts
py_kpack='pack-kaldo.py'
py_mdpack='pack-md.py'
py_path='path-gen.py'
py_final="path-compare.py"

# Data files
tarball="forces.tgz"
forcedir="forces"

# mdin="md.in.band_path" # Option 1
mdin="md.in.coord_path" # Option 2
mdout="md.out.txt"
mdpack="md.out.pack"
mdcomp="md.out.npy"

kcomp="kaldo.out.npy"
kpack="kaldo.out.pack"

ptxt="path.out.txt"

final="mismatch.txt"
finalout="mismatch.npy"

# executables
matdyn="/home/nwlundgren/develop/quantumespresso/build-6.21.2023/bin/matdyn.x"

printf "\n\n!!\tWorking on comparisons \n"
#########################################################################
##############  E  S  P  R  E  S  S  O  ########################
if [ ! -f ${mdcomp} ];
then
  printf "! Attempting to package QE data ..."
  if [ ! -f ${forcedir}/espresso.ifc2 ]; # untar
  then
    printf "\n\tUntar forces to ${forcedir} .."
    tar -xvzf ${tarball}
    printf "\tdone!"
  fi

  printf "\n\tMake text file for k-path at ${ptxt} .."
  python ${py_path}
  printf "\tdone!"

  printf "\n\tGenerating matdyn input ${mdin} from ${mdin}.tmp .."
  cat ${mdin}.tmp > ${mdin}
  printf "\n$( cat ${ptxt} | wc -l )\n" >> ${mdin}
  cat ${ptxt} >> ${mdin}
  printf "\tdone!"

  printf "\n\tRunning matdyn.x from path ${matdyn}.."
  ${matdyn} < ${mdin} &> ${mdout}
  printf "\tdone!"
  if [ $( wc -l < ${mdout}) -lt 5000 ];
  then # if the output is small something went wrong
    printf "\n\t\tUnusual matdyn.x behaviour, fatal error\n\n"
    exit 1
  fi

  python ${py_mdpack} > ${mdpack}
  printf "\nMatdyn Segment Complete! :) \n"
fi
################################################################



##############  k  A  L  D  O  #################################
if [ ! -f ${kcomp} ];
then
  printf "! Attempting to package comparable kALDo data ..."
  if [ ! -f ${mdcomp} ];
  then
    printf "\n\tMissing matdyn output, no q-pt list to compare\n\n"
    exit 1
  fi

  printf "\n\tRunning kaldo .."
  python ${py_kpack} &> ${kpack}
  printf "\tdone!"
  printf "\n\tKaldo data was packed!"
  printf "Kaldo Segment Complete! :) \n"
fi

################################################################
########################################################################

printf "\n! We think data for both programs has been output and packaged uniformly\n"
printf "Attempting to run comparison script .. "

# Attempt to run comparison
python ${py_final} > ${final}

printf "\tComparison has run and exited!\n\tOutput can be found at out.py.comparison\n"
printf "\tMismatches saved to ${finalout}\n"
printf "\n\n"
exit 0






