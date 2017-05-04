package com.rho.neural

object basics {

  import com.rho.neural.types.{Signals, Real, sum_vectors, mat_product}

  /* A neuron is an object that processes a collection of Signals
   * through weights, bias. 
   * The activator function is implemented in the Network class */
  case class Neuron(val w: Signals, val b: Real) {
    def apply(x: Signals): Real = {
      if (w.size != x.size) throw new Error("weights and x must be same size")
      else (w zip x).map(z => z._1*z._2).reduce(_+_) + b
    }
    override def toString(): String = {
      val ws = w.foldLeft[String]("")((a,b) => if (a.isEmpty) b.toString else a+","+b.toString)
      "Neuron[weights=("+ws+"),bias="+b+"]"
    }
    def +(other: Neuron): Neuron = {
      new Neuron(sum_vectors(this.w,other.w), this.b+other.b)
    }
  }

  /* Neuron layer class */
  case class Layer(init: Neuron*) {
    val neurons = Array(init: _*)
    val weights = neurons.map(_.w)
    val biases  = neurons.map(_.b)
    def apply(x: Signals): Signals = sum_vectors(mat_product(weights,x),biases)
    override def toString(): String = {
      def concat(s:String)(a:String,b:String): String = if (a.isEmpty) b else a+s+b
      neurons.foldLeft[String]("")((a,b)=>concat(",")(a,b.toString))
    }
  }

}