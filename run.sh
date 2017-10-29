#!/bin/bash

disclaimer="
            --> Kitt<3ns Recommendation ENGINE <--

    Capabilities:

    Random Recommendations
    Top Listened Recommendations
    Top Included Recommendations
    Top Tags Recommendations
    TF-IDF Recommendations
    Various combined methods
    Test Mode
    Script Mode

    Further informations available on the README

    Please select your choice wisely...
"

running="
         Running...
"

end="
         Completed!
         Result writed to file correctly
         Bye
"

end_test="
         Test Mode Completed!
         Average Accuracy
"

command -v pypy >/dev/null
if [[ "$?" == "1" ]]; then
  whiptail --yesno "pypy is required, proceed with installation?" 10 40
  if [[ "$?" == "0" ]]; then
    sudo apt install pypy
  else
    exit 0
  fi
fi

mem=$(free | awk 'FNR == 2 {print int($7/(1024*1024*1.5))}')

if [[ "$mem" == "0" ]]; then
  whiptail --msgbox "Sorry, the system does not have enough memory available (1.5Gb minimum is required)" 10 50
  exit 0
fi

cpu=$(nproc --all)

core=$(($mem<$cpu?$mem:$cpu))

whiptail --msgbox "$disclaimer" 20 70

mode=$(whiptail --menu "Select Operational Mode" 20 70 5 \
0 "Test Mode" \
1 "Recommendation Mode" \
2 "Script Mode" \
3 "Debug Mode" \
3>&2 2>&1 1>&3)
if [[ -z "$mode" ]]; then exit 0; fi

case $mode in
    2)
        script=$(whiptail --menu "Which script do you want to run?" 20 80 5 \
        0 "Compute test set (x3)" 3>&2 2>&1 1>&3)
        if [[ -z "$script" ]]; then exit 0; fi
        /usr/bin/pypy scripts/test_set_generator.py
    ;;
    *)
        recommendations=$(whiptail --menu "Select Recommendations Method" 20 70 10 \
        0 "Random" \
        1 "Top Listened" \
        2 "Top Included" \
        3 "Top Tags" \
        4 "TF-IDF based on tags" \
        5 "Top-Tag combined TfIdf" \
        6 "TfIdf combined Top-Tag" \
        7 "TfIdf based on titles" \
        8 "Tfidf tags combined tfIdf titles" \
        9 "Top-Tag combined TfIdf titles" \
        10 "Tfidf Titles filter applied on normal tfidf" \
        11 "Bad tfidf recs" \
        12 "Artist Recommendations + tfidf padding" \
        13 "Hybrid recommendations" \
        14 "Neighborhood similarity recommendations" \
        16 "Naive Bayes computation" \
        3>&2 2>&1 1>&3)

        if [[ -z "$recommendations" ]]; then exit 0; fi

        if [[ "$mode" == "3" ]]
        then /usr/bin/pypy kittens.py 0 "$recommendations" 1 1
        else /usr/bin/pypy kittens.py "$mode" "$recommendations" "$core" 1 | whiptail --gauge "$running" 15 60 0
        fi
    ;;
esac
case $mode in
    0)
        whiptail --textbox data/test_result1.csv 12 60
    ;;
    3)
        exit 0
    ;;
    *)
        whiptail --msgbox "$end" 20 50
    ;;
esac
exit 0
