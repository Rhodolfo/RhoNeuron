package com.rho

object neurons {

  // Custom types
  type Real = Double
  type Signals = Array[Real]
  def Signals(xs: Real*) = Array(xs: _*)
  type Layer = Array[Neuron]
  def Layer(xs: Neuron*) = Array(xs: _*)

  /* A neuron is an object that processes a collection of Signals
   * through weights, bias and an activator function */
  abstract class Neuron(w: Signals, b: Real) {
    def argument(x: Signals): Real = {
      if (w.size != x.size) throw new Error("weights and x must be same size")
      else (w zip x).map(z => z._1*z._2).reduce(_+_) + b
    }
    def activator(z: Real): Real
    def apply(x: Signals): Real = activator(argument(x))
    override def toString(): String = {
      val ws = w.foldLeft[String]("")((a,b) => if (a.isEmpty) b.toString else a+","+b.toString)
      "Neuron[weights=("+ws+"),bias="+b+"]"
    }
  }

  // Perceptron is just a 0-1 activator
  case class Perceptron(w:Signals, b: Real) extends Neuron(w,b) {
    def activator(z: Real): Real = if (z>0) 1 else 0
  }

  // Sigmoid function to make Perceptrons differentiable
  case class Sigmoid(w:Signals, b: Real) extends Neuron(w,b) {
    def activator(z: Real): Real = 1/(1+scala.math.exp(-z))
  }

  /* Differentiable trait */
  trait Differentiable {
    def prepare(x: Signals): List[Real]
    def backProp(x: Signals): List[Real]
  }

  /* A neural network is a collection of Neuron Layers,
   * it can process Signals, the response Signal is the head of the output of forwardProp */
  class Network(layers: List[Layer]) {

    override def toString(): String = {
      def  concat(s:String)(a:String,b:String): String = if (a.isEmpty) b else a+s+b
      def layer2string(layer: Layer): String = layer.foldLeft[String]("")((a,b)=>concat(",")(a,b.toString))
      val ls = layers.foldLeft[String]("")((a,b) => concat("\n")(a,layer2string(b)))
      "Network of "+layers.size+ " layers:\n"+ls
    }

    def forwardProp(x: Signals): List[Signals] = {
      def build(net: List[Layer], acc: List[Signals]): List[Signals] = {
        if (net.isEmpty) acc
        else build(net.tail, net.head.map(neuron => neuron apply acc.head) :: acc)
      }
      build(layers,List(x))
    }

    def predict(x: Signals): Signals = {
      val signals = forwardProp(x)
      signals.head
    }

  }

}
