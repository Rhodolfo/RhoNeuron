import org.scalatest.FunSuite

class NeuralTesting extends FunSuite {

  import com.rho.neurons._

  test("Single input Perceptron") {
    val weights = Signals(1.0)
    val neuron = new Perceptron(weights,2.0)
    assert(1===neuron(Signals(3.0)))
    assert(0===neuron(Signals((-2.0))))
  }

  test("Single input Sigmoid") {
    val weights = Signals(1.0)
    val neuron = new Sigmoid(weights,2.0)
    assert(0.5===neuron(Signals((-2.0))))
    assert(0.5<neuron(Signals((2.0))))
    assert(0.5>neuron(Signals((-5.0))))
  }

  test("XOR Gate, Perceptron Network") {
    val top = new Perceptron(Signals(0.6,0.6),-1.0)
    val bot = new Perceptron(Signals(1.1,1.1),-1.0)
    val out = new Perceptron(Signals(-2.0,1.1),-1.0)
    val layers = List(Layer(top,bot),Layer(out))
    val network = new Network(layers)
    assert(0===network.backSweep(Signals(0.0,0.0)).head.head)
    assert(1===network.backSweep(Signals(1.0,0.0)).head.head)
    assert(1===network.backSweep(Signals(0.0,1.0)).head.head)
    assert(0===network.backSweep(Signals(1.0,1.0)).head.head)
  }

  test("XOR Gate, Sigmoid Network") {
    val top = new Sigmoid(Signals(1.0,1.0),0.0)
    val bot = new Sigmoid(Signals(2.0,2.0),0.0)
    val out = new Sigmoid(Signals(-1000.0,850.0),0.0)
    val layers = List(Layer(top,bot),Layer(out))
    val network = new Network(layers)
    assert(0.25>network.backSweep(Signals(0.0,0.0)).head.head)
    assert(0.75<network.backSweep(Signals(1.0,0.0)).head.head)
    assert(0.75<network.backSweep(Signals(0.0,1.0)).head.head)
    assert(0.25>network.backSweep(Signals(1.0,1.0)).head.head)
  }

}
