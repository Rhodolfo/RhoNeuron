package com.rho.neural

object types {

  // Custom types
  type Real = Double
  type Signals = Array[Real]
  type Throughput = Array[(Real,Real)]
  def Signals(xs: Real*) = Array(xs: _*)

  // Auiliary operations
  def had_product(x: Signals, y: Signals): Signals = {
    if (x.size == y.size) (x zip y).map(z => z._1*z._2)
    else throw new Error("Signals must be equal size, "+x.size+"!="+y.size)
  }
  def dot_product(x: Signals, y: Signals): Real = {
    if (x.size == y.size) had_product(x,y).reduce(_+_)
    else throw new Error("Signals must be equal size, "+x.size+"!="+y.size)
  }
  def mat_product(m: Array[Signals], x: Signals): Signals = m.map(dot_product(_,x))
  def sum_vectors(x: Signals, y: Signals): Signals = {
    if (x.size == y.size) (x zip y).map(z => z._1+z._2)
    else throw new Error("Signals must be equal size, "+x.size+"!="+y.size)
  } 
  def sum_matrices(x: Array[Signals], y: Array[Signals]): Array[Signals] = {
    if (x.size == y.size) (x zip y).map(z => sum_vectors(z._1,z._2))
    else throw new Error("Matrices must be equal size, "+x.size+"!="+y.size)
  }

}
