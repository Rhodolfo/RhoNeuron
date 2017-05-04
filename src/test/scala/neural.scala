import org.scalatest.FunSuite

class NeuralTesting extends FunSuite {

  import com.rho.neural.types.{Signals}
  import com.rho.neural.basics.{Neuron, Layer}
  import com.rho.neural.networks.{PerceptronNetwork, SigmoidNetwork}

  test("XOR Gate, Perceptron Network") {
    val top = new Neuron(Signals(0.6,0.6),-1.0)
    val bot = new Neuron(Signals(1.1,1.1),-1.0)
    val out = new Neuron(Signals(-2.0,1.1),-1.0)
    val layers = Array(Layer(top,bot),Layer(out))
    val network = new PerceptronNetwork(layers)
    assert(0===network.predict(Signals(0.0,0.0)).head)
    assert(1===network.predict(Signals(1.0,0.0)).head)
    assert(1===network.predict(Signals(0.0,1.0)).head)
    assert(0===network.predict(Signals(1.0,1.0)).head)
  }

  test("XOR Gate, Sigmoid Network") {
    val top = new Neuron(Signals(1.0,1.0),0.0)
    val bot = new Neuron(Signals(2.0,2.0),0.0)
    val out = new Neuron(Signals(-1000.0,850.0),0.0)
    val layers = Array(Layer(top,bot),Layer(out))
    val network = new SigmoidNetwork(layers)
    assert(0.25>network.predict(Signals(0.0,0.0)).head)
    assert(0.75<network.predict(Signals(1.0,0.0)).head)
    assert(0.75<network.predict(Signals(0.0,1.0)).head)
    assert(0.25>network.predict(Signals(1.0,1.0)).head)
  }

  test("Deltas") {
    /*
    val top = new Neuron(Signals(1.0,1.0),0.0)
    val bot = new Neuron(Signals(2.0,2.0),0.0)
    val out = new Neuron(Signals(-1000.0,850.0),0.0) 
    */
    val top = new Neuron(Signals(1.0,-7.0),0.0)
    val bot = new Neuron(Signals(6.0,-4.0),0.0)
    val out = new Neuron(Signals(-1.0,23.0),0.0)
    val layers = Array(Layer(top,bot),Layer(out))
    val network = new SigmoidNetwork(layers) 
    val list = List(
      (Signals(0.0,0.0), Signals(0.0)),
      (Signals(1.0,0.0), Signals(1.0)),
      (Signals(0.0,1.0), Signals(1.0)),
      (Signals(1.0,1.0), Signals(0.0))
    )
    list.foreach { pair => pair match {
        case (input, target) => {
          println("iter")
          val output = network.predict(input)
          val (cost,deltas) = network.deltas(input,target)
          output.foreach { println }
          deltas.foreach {z =>
            z.foreach {zz => 
              print("|"+zz+"|")
            }
            println(" ")
          }
          println(cost)
        }
      }
    }
  }

}
