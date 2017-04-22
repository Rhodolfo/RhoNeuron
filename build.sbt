// Common settings
lazy val commonSettings = Seq(
  libraryDependencies += "org.scalactic" %% "scalactic" % "3.0.1",
  libraryDependencies += "org.scalatest" %% "scalatest" % "3.0.1" % "test",
  organization := "com.rho",
  version := "0.0.1",
  scalacOptions := Seq("-feature"),
  scalaVersion := "2.12.1")

// JSON parsing library
// lazy val json = Seq(libraryDependencies += "net.liftweb" %% "lift-json" % "3.0.1")

lazy val neural = (project in file("."))
  .settings(commonSettings: _*)
  .settings(name := "NeuralNetwork")
