import org.scalatest.FunSuite

class NeuralTesting extends FunSuite {

  import com.rho.neural.types.{Signals}
  import com.rho.neural.basics.{Neuron, Layer}
  import com.rho.neural.networks.{PerceptronNetwork, SigmoidNetwork}

  test("XOR Gate, Perceptron Network") {
    val layer1 = Array( (Array(0.6,0.6),-1.0), (Array(1.1,1.1),-1.0) )
    val layer2 = Array( (Array(-2.0,1.1),-1.0) )
    val layers = Array(Layer(layer1),Layer(layer2))
    val network = new PerceptronNetwork(layers)
    assert(0===network.predict(Signals(0.0,0.0)).head)
    assert(1===network.predict(Signals(1.0,0.0)).head)
    assert(1===network.predict(Signals(0.0,1.0)).head)
    assert(0===network.predict(Signals(1.0,1.0)).head)
  }

  trait XOR_Gate {
    val layer1 = Array( (Array(1.0,1.0),0.0), (Array(2.0,2.0),0.0) )
    val layer2 = Array( (Array(-1000.0,850.0),0.0) )
    val layers = Array(Layer(layer1),Layer(layer2))
    val network = new SigmoidNetwork(layers) 
    val trainData = List(
      (Signals(0.0,0.0), Signals(0.0)),
      (Signals(1.0,0.0), Signals(1.0)),
      (Signals(0.0,1.0), Signals(1.0)),
      (Signals(1.0,1.0), Signals(0.0))
    )
  }

  new XOR_Gate {
    test("XOR Gate, Sigmoid Network") {
      val epsilon = 0.01
      val delta = 0.01
      trainData.foreach {p =>
        p match {
          case (input,target) => {
            val res = network.predict(input).head
            val pre = target.head
            assert(pre-epsilon < res && pre+epsilon > res)
          }
        } 
      }
      val (terr, _, _) = network.sumGradients(trainData)
      assert(terr<delta)
    }
  }
  
  test("Deltas") {
    val badlayer1 = Array( (Signals(1.0,-7.0),0.0), (Signals(6.0,-4.0),0.0) )
    val badlayer2 = Array( (Signals(-1.0,23.0),0.0) ) 
    val badlayers = Array(Layer(badlayer1),Layer(badlayer2))
    val badnetwork = new SigmoidNetwork(badlayers) 
  }

}
