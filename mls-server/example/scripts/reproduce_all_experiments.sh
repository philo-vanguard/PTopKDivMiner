#!/bin/bash

echo -e "------------ vary all for airports ------------"
./exp_airports.sh

echo -e "------------ vary all for hospital ------------"
./exp_hospital.sh

echo -e "------------ vary all for dblp ------------"
./exp_dblp.sh

echo -e "------------ vary all for inspection ------------"
./exp_inspection.sh

echo -e "------------ vary all for ncvoter ------------"
./exp_ncvoter.sh
