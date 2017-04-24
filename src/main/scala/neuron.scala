package com.rho

object neurons {

  // Custom types
  type Real = Double
  type Signals = Array[Real]
  def Signals(xs: Real*) = Array(xs: _*)
  type Layer = Array[Neuron]
  def Layer(xs: Neuron*) = Array(xs: _*)

  // Basic neuron classes
  abstract class Neuron(w: Signals, b: Real) {
    def argument(x: Signals): Real = {
      if (w.size != x.size) throw new Error("weights and x must be same size")
      else (w zip x).map(z => z._1*z._2).reduce(_+_) + b
    }
    def process(z: Real): Real
    def apply(x: Signals): Real = process(argument(x))
    override def toString(): String = {
      val ws = w.foldLeft[String]("")((a,b) => if (a.isEmpty) b.toString else a+","+b.toString)
      "Neuron[weights=("+ws+"),bias="+b+"]"
    }
  }

  class Perceptron(w:Signals, b: Real) extends Neuron(w,b) {
    def process(z: Real): Real = if (z>0) 1 else 0
  }

  class Sigmoid(w:Signals, b: Real) extends Neuron(w,b) {
    def process(z: Real): Real = 1/(1+scala.math.exp(-z))
  }

  // Neural network
  class Network(layers: List[Layer]) {

    override def toString(): String = {
      def  concat(s:String)(a:String,b:String): String = if (a.isEmpty) b else a+s+b
      def layer2string(layer: Layer): String = layer.foldLeft[String]("")((a,b)=>concat(",")(a,b.toString))
      val ls = layers.foldLeft[String]("")((a,b) => concat("\n")(a,layer2string(b)))
      "Network of "+layers.size+ " layers:\n"+ls
    }

    def backSweep(x: Signals): List[Signals] = {
      def build(net: List[Layer], acc: List[Signals]): List[Signals] = {
        if (net.isEmpty) acc
        else build(net.tail, net.head.map(neuron => neuron apply acc.head) :: acc)
      }
      build(layers,List(x))
    }


  }

}
