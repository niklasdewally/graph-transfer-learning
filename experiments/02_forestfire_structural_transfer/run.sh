#!/bin/bash

RES=1
until [ "$RES" == "0" ]
do
	python3 ./run.py
	RES=$?
done

