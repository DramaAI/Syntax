import unittest


from modules.loss import TestLossModules
from modules.gpt import TestGPTModules


if __name__ == "__main__":
    loader = unittest.TestLoader()
    suit = unittest.TestSuite()


    suit.addTest(loader.loadTestsFromTestCase(TestLossModules))
    suit.addTest(loader.loadTestsFromTestCase(TestGPTModules))

    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suit)