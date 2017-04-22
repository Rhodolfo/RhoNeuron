import org.scalatest.FunSuite

class DiskTesting extends FunSuite {

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

}
