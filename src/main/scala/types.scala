package com.rho.neural

object types {

  // Custom types
  type Real = Double
  type Signals = Array[Real]
  type Throughput = Array[(Real,Real)]
  def Signals(xs: Real*) = Array(xs: _*)

  // Auiliary operations
  def had_product(x: Signals, y: Signals): Signals = (x zip y).map(z => z._1*z._2)
  def dot_product(x: Signals, y: Signals): Real = had_product(x,y).reduce(_+_)
  def mat_product(m: Array[Signals], x: Signals): Signals = m.map(dot_product(_,x))
  def sum_vectors(x: Signals, y: Signals): Signals = (x zip y).map(z => z._1+z._2)

}
