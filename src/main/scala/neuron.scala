package com.rho

object neurons {

  // Custom types
  type Real = Double
  type Signals = List[Double]
  def Signals(xs: Real*) = List(xs: _*)
  type Layer = List[Neuron]

  // Basic neuron classes
  abstract class Neuron(w: Signals, b: Real) {
    def argument(x: Signals): Real = {
      if (w.size != x.size) throw new Error("weights and x must be same size")
      else (w zip x).map(z => z._1*z._2).reduce(_+_) + b
    }
    def process(z: Real): Real
    def apply(x: Signals): Real = process(argument(x))
  }

  class Perceptron(w:Signals, b: Real) extends Neuron(w,b) {
    def process(z: Real): Real = if (z>0) 1 else 0
  }

  class Sigmoid(w:Signals, b: Real) extends Neuron(w,b) {
    def process(z: Real): Real = 1/(1+scala.math.exp(-z))
  }

  // Neural network
  class Network(network: List[Layer]) {

    def backSweep(x: Signals): List[Signals] = {
      def build(net: List[Layer], acc: List[Signals]): List[Signals] = {
        if (net.isEmpty) acc
        else build(net.tail, net.head.map(neuron => neuron apply acc.head) :: acc)
      }
      build(network,List(x))
    }

  }

}
