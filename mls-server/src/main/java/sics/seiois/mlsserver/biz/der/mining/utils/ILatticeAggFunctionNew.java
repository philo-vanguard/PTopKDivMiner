package sics.seiois.mlsserver.biz.der.mining.utils;

import org.apache.spark.api.java.function.Function2;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class ILatticeAggFunctionNew implements Function2<Lattice, Lattice, Lattice> {
   private static Logger log = LoggerFactory.getLogger(ILatticeAggFunctionNew.class);

   @Override
    public Lattice call(Lattice mg_1, Lattice mg_2) {

       if (mg_1 == null && mg_2 == null) {
           return new Lattice();
       }

       if (mg_1 == null) {
           return mg_2;
       }

       if (mg_2 == null) {
           return mg_1;
       }

       Lattice allLattice = new Lattice(mg_1.getMaxTupleNumPerRule());

       for (LatticeVertex lv : mg_1.getLatticeLevel().values()) {
           allLattice.addLatticeVertexNew(lv);
       }
       for (LatticeVertex lv : mg_2.getLatticeLevel().values()) {
           allLattice.addLatticeVertexNew(lv);
       }

       return allLattice;

   }
}
