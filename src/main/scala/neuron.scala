package com.rho

object neurons {

  // Custom types
  type Real = Double
  type Signals = Array[Real]
  type Throughput = Array[(Real,Real)]
  def Signals(xs: Real*) = Array(xs: _*)



  /* A neuron is an object that processes a collection of Signals
   * through weights, bias. 
   * The activator function is implemented in the Network class */
  class Neuron(val w: Signals, val b: Real) {
    def apply(x: Signals): Real = {
      if (w.size != x.size) throw new Error("weights and x must be same size")
      else (w zip x).map(z => z._1*z._2).reduce(_+_) + b
    }
    override def toString(): String = {
      val ws = w.foldLeft[String]("")((a,b) => if (a.isEmpty) b.toString else a+","+b.toString)
      "Neuron[weights=("+ws+"),bias="+b+"]"
    }
  }

  def had_product(x: Signals, y: Signals): Signals = (x zip y).map(z => z._1*z._2)
  def dot_product(x: Signals, y: Signals): Real = had_product(x,y).reduce(_+_)
  def mat_product(m: Array[Signals], x: Signals): Signals = m.map(dot_product(_,x))
  def sum_vectors(x: Signals, y: Signals): Signals = (x zip y).map(z => z._1+z._2)

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



  /* A neural network is a collection of Neuron Layers,
   * it can process Signals, the response Signal is the head of the output of forwardProp */
  abstract class Network(val layers: List[Layer]) {

    def activator(x: Real): Real // Abstract method

    def forwardProp(x: Signals): List[Throughput] = {
       def build(net: List[Layer], acc: List[Throughput]): List[Throughput] = {
        if (net.isEmpty) acc
        else {
          val sign  = acc.head.unzip._2
          val zeta  = net.head apply sign
       // val zeta  = net.head.neurons.map(neuron => neuron apply sign)
          val sigma = zeta map activator
          build(net.tail, (zeta zip sigma) :: acc)
        }
      }
      val init = x zip x // Initial signal is just x mapped to x
      build(layers,List(init))    
    }

    def predict(x: Signals): Signals = {
      val signals = forwardProp(x)
      signals.head.unzip._2
    }

    def cost(x: Signals, y: Signals): Real

    override def toString(): String = {
      def concat(s:String)(a:String,b:String): String = if (a.isEmpty) b else a+s+b
      val ls = layers.foldLeft[String]("")((a,b) => concat("\n")(a,b.toString))
      "Network of "+layers.size+ " layers:\n"+ls
    }

  }

  /* More methods for differentiable activators */
  trait Differentiable {
    def dactivator_dargument(x: Real): Real 
    def dcost_dsignal(x: Signals, target: Signals): Signals
  }

  /* Quadratic cost function */
  trait QuadraticCost {
    def quadCost(x: Signals, y: Signals, f: Signals => Signals): Real = {
      import scala.math.pow
      0.5*sum_vectors(f(x),y.map(-_)).reduce(pow(_,2)+pow(_,2))
    }
  }

  /* Perceptron */
  class PerceptronNetwork(layers: List[Layer]) extends Network(layers) with QuadraticCost {
    def activator(x: Real): Real = if (x>0) 1 else 0
    def cost(x: Signals, target: Signals): Real = quadCost(x,target,predict)
  }

  /* Sigmoid */
  class SigmoidNetwork(layers: List[Layer]) extends Network(layers) 
  with Differentiable with QuadraticCost {
    def activator(x: Real): Real = 1 / (1 + scala.math.exp(-x))
    def dactivator_dargument(x: Real): Real = activator(x)*(1 - activator(x))
    def cost(x: Signals, target: Signals): Real = quadCost(x,target,predict)
    def dcost_dsignal(a: Signals, target: Signals): Signals = sum_vectors(a,target.map(-_))

    /* Steepest descent */
    def deltas(x: Signals, target: Signals): List[Signals] = {
      def iter(azList: List[Throughput], wList: List[Array[Signals]], acc: List[Signals], 
      pastDelta: Signals, flag: Boolean = false): List[Signals] = {
        if (azList.isEmpty) acc
        else {
          val (aL,zL) = azList.head.unzip
          val wL = wList.head
          val factor = {
            if (flag) dcost_dsignal(aL,target)
            else mat_product(wL.transpose,pastDelta)
          }
          val sigp = zL map dactivator_dargument
          val prod = had_product(factor,sigp)
          iter(azList.tail, wList.tail, List(prod) ++ acc, prod)
        }
      }
      iter(forwardProp(x),layers.map(_.weights),Nil,Array(),true)
    }

  }

}
