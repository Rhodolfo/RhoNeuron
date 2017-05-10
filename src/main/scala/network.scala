package com.rho.neural

object networks {

  import com.rho.neural.types.{Real, Signals, Throughput, sum_vectors, mat_product, had_product}
  import com.rho.neural.basics.{Neuron, Layer}

  /* A neural network is a collection of Neuron Layers,
   * it can process Signals, the response Signal is the head of the output of forwardProp */
  abstract class Network(val layers: Array[Layer]) {

    def activator(x: Real): Real // Abstract method

    def forwardProp(x: Signals): Array[Throughput] = {
       def build(net: Array[Layer], acc: Array[Throughput]): Array[Throughput] = {
        if (net.isEmpty) acc
        else {
          val sign  = acc.last.unzip._2
          val zeta  = net.head apply sign
          val sigma = zeta map activator
          build(net.tail, acc ++ Array(zeta zip sigma))
        }
      }
      val init = x zip x // Initial signal is just x mapped to x
      build(layers,Array(init))    
    }

    def predict(x: Signals): Signals = {
      val signals = forwardProp(x)
      signals.last.unzip._2
    }

    def cost(prediction: Signals, target: Signals): Real

    override def toString(): String = {
      def concat(s:String)(a:String,b:String): String = if (a.isEmpty) b else a+s+b
      val ls = layers.foldLeft[String]("")((a,b) => concat("\n")(a,b.toString))
      "Network of "+layers.size+ " layers:\n"+ls
    }

  }



  /* Method for trainable networks */
  trait Trainable {
    def train(throughputs: Traversable[(Signals,Signals)], 
      tolerance: Real = 1e-2, maxIter: Int = 100): Network
  }



  /* More methods for differentiable activators */
  trait Differentiable {
    def dactivator_dargument(x: Real): Real 
    def dcost_dsignal(params: Signals, target: Signals): Signals
  }



  /* Quadratic cost function */
  trait QuadraticCost {
    def quadCost(prediction: Signals, target: Signals): Real = {
      import scala.math.pow
      0.5*sum_vectors(prediction,target.map(-_)).map(pow(_,2)).sum
    }
  }


  /* Steepest descent */ 
  abstract class SteepestDescent(layers: Array[Layer]) extends Network(layers) with Differentiable {

    /* Calculation of gradients by biases and weights
     * aL = output of layer L
     * wL = weights of layer L
     * bL = biases of layer L
     * zL = wL*a(L-1) + bL
     * deltaL = dC/dzL
     * dC/dbL = deltaL
     * dC/dwL = a(L-1)*deltaL
    */
    def computeGradients(input: Signals, target: Signals): 
    (Real,Array[Signals],Array[Array[Signals]]) = {
      def iter(azArray: Array[Throughput], wArray: Array[Array[Signals]], acc: Array[Signals], 
      flag: Boolean = false): Array[Signals] = {
        if (azArray.isEmpty) acc
        else {
          val (aL,zL) = azArray.last.unzip
          val wL = wArray.last
          val pastDelta = if (acc.isEmpty) Signals(0.0) else acc.head
          val factor = {
            if (flag) dcost_dsignal(aL,target)
            else mat_product(wL.transpose,pastDelta)
          }
          val sigp = zL map dactivator_dargument
          val prod = had_product(factor,sigp)
          if (flag) iter(azArray.init, wArray, Array(prod))
          else iter(azArray.init, wArray.init, Array(prod) ++ acc)
        }
      }
      val matrix = layers.map(_.weights)
      val throughputs = forwardProp(input)
      val tcost  = cost(throughputs.last.unzip._2,target)
      val deltas = iter(throughputs.tail,matrix,Array(),true)
      val weightGradient = (deltas zip throughputs.init.map(_.unzip._1)).map(
        (pair: (Signals,Signals)) => pair match {
          case(deltaL: Signals, arrayL: Signals) => 
            deltaL.map(delta => arrayL.map(_*delta))
        }
      )
      if (throughputs.tail.size==matrix.size) (tcost,deltas,weightGradient)
      else throw new Error("Something went wrong")
    }

    def train(trainData: List[(Signals,Signals)],
    eta: Real = 1e-1, tolerance: Real = 1e-2, maxIter: Int = 100): Network = {
      def singleTarget(input: Signals, target: Signals): (Real,Array[Signals],Array[Array[Signals]]) = {
        computeGradients(input,target)
      }
      ???
    }
 

  }



  /* Perceptron */
  class PerceptronNetwork(layers: Array[Layer]) extends Network(layers) with QuadraticCost {
    def activator(x: Real): Real = if (x>0) 1 else 0
    def cost(prediction: Signals, target: Signals): Real = quadCost(prediction,target)
  }



  /* Sigmoid */
  class SigmoidNetwork(layers: Array[Layer]) extends SteepestDescent(layers) with QuadraticCost {
    // Defining basic abstract methods
    def activator(x: Real): Real = 1 / (1 + scala.math.exp(-x))
    def dactivator_dargument(x: Real): Real = activator(x)*(1 - activator(x))
    def cost(prediction: Signals, target: Signals): Real = quadCost(prediction,target)
    def dcost_dsignal(a: Signals, target: Signals): Signals = sum_vectors(a,target.map(-_))
  }



}
