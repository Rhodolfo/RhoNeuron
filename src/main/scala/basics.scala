package com.rho.neural

object basics {

  import com.rho.neural.types.{Signals, Real, sum_vectors, sum_matrices, mat_product}

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
  case class Layer(val weights: Array[Signals], val biases: Signals) {
    def this(pair: Array[(Signals,Real)]) = this(pair.unzip._1,pair.unzip._2)
    def size(): Int = {
      if (weights.size==biases.size) weights.size 
      else throw new Error("Weight matrix and biases must be compatible size")
    }
    def *(scalar: Real) = {
      Layer(weights.map(row => row.map(_*scalar)), biases.map(_*scalar))
    }
    def +(other: Layer) = {
      if (this.size==other.size) {
        Layer(sum_matrices(this.weights,other.weights), sum_vectors(this.biases, other.biases))
      } else {
        throw new Error("Layers must be same size to sum")
      }
    }
    def apply(x: Signals): Signals = sum_vectors(mat_product(weights,x),biases)
    override def toString(): String = {
      "Stub method. Layer of size "+size
    }
  }

  object Layer {
    def apply(pair: Array[(Signals,Real)]) = new Layer(pair)
  }


}
