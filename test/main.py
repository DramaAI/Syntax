import unittest


from modules.loss import TestLossModules
from test.modules.bert import TestGPTModules


if __name__ == "__main__":
    loader = unittest.TestLoader()
    suit = unittest.TestSuite()


    suit.addTest(loader.loadTestsFromTestCase(TestLossModules))
    suit.addTest(loader.loadTestsFromTestCase(TestGPTModules))
    print(f"\nUnit Test 🧪\n{'':-^70}")
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suit)