package HomeWork1;

import com.sun.tools.doclets.formats.html.SourceToHTMLConverter;
import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

public class LinearRegression implements Classifier {
	
    private int m_ClassIndex;
	private int m_truNumAttributes;
	private double[] m_coefficients;
	private double m_alpha;

	private final double ERROR_REQUIRED = 0.003;
	private final int MIN_ALPHA = -17;
	
	//the method which runs to train the linear regression predictor, i.e.
	//finds its weights.
	@Override
	public void buildClassifier(Instances trainingData) throws Exception {
		m_ClassIndex = trainingData.classIndex();
		m_truNumAttributes = trainingData.numAttributes();
		m_coefficients = new double[m_truNumAttributes + 1];

		if (m_alpha == 0)
		    findAlpha(trainingData);

        boolean errorLargerThanRequired = true;
        int numberOfIterations = 0;
        double error, newError;
        error = Double.MAX_VALUE;
        m_coefficients = initCoefficients(m_coefficients);

        while (errorLargerThanRequired) {
            numberOfIterations++;
            m_coefficients = gradientDescent(trainingData);
            if ((numberOfIterations % 100) == 0) {
                newError = calculateMSE(trainingData);
                if (Math.abs(error - newError) < ERROR_REQUIRED)
                    errorLargerThanRequired = false;
                else
                    error = newError;
            }
        }

    }
	
	private void findAlpha(Instances data) throws Exception {
		double currentAlphaBestError, currentError, totalBestError;
		int bestAlphaIndex = MIN_ALPHA;

		currentAlphaBestError = Double.MAX_VALUE;
		totalBestError = Double.MAX_VALUE;

		for (int i = MIN_ALPHA; i < 0; i++) {
            m_alpha = Math.pow(3, i);
            m_coefficients = initCoefficients(m_coefficients);
            for (int j = 0; j < 20000; j++) {
                m_coefficients = gradientDescent(data);
                if ((j % 100) == 0) {
                    currentError = calculateMSE(data);
                    if (currentError < currentAlphaBestError)
                        currentAlphaBestError = currentError;
                    else
                        break;
                }
            }

            if (currentAlphaBestError < totalBestError) {
                totalBestError = currentAlphaBestError;
                bestAlphaIndex = i;
            }
        }
        m_alpha = Math.pow(3, bestAlphaIndex);
    }

    private double[] initCoefficients(double[] coefficients) {

	    for (int i = 0; i < coefficients.length; i++)
            coefficients[i] = 1;
	    return coefficients;
    }

    /**
	 * An implementation of the gradient descent algorithm which should
	 * return the weights of a linear regression predictor which minimizes
	 * the average squared error.
     * 
	 * @param trainingData
	 * @throws Exception
	 */
	private double[] gradientDescent(Instances trainingData)
			throws Exception {
        double[] newCoefficients = new double[m_truNumAttributes + 1];
        for (int i = 0; i < newCoefficients.length; i++) {
            newCoefficients[i] = updateSingleCoefficient(trainingData, m_coefficients, i);
        }
        return newCoefficients;

	}

    private double updateSingleCoefficient(Instances trainingData, double[] coefficients, int coefficientIndex) {
        double currentPartialDerivative, sumPartialDerivative, newSingleCoefficient;
        Instance currentInstance;

        sumPartialDerivative = 0;

        for (int i = 0; i < trainingData.numInstances(); i++) {
            currentInstance = trainingData.instance(i);
            currentPartialDerivative = instanceCoefficientsInnerProduct(currentInstance, coefficients);
            currentPartialDerivative -= currentInstance.classValue();
            if (coefficientIndex != 0) {
                currentPartialDerivative *= currentInstance.value(coefficientIndex - 1);
            }
            sumPartialDerivative += currentPartialDerivative;
        }

        newSingleCoefficient = coefficients[coefficientIndex] -
                (m_alpha * sumPartialDerivative / trainingData.numInstances());

        return newSingleCoefficient;
	}

    private double instanceCoefficientsInnerProduct(Instance currentInstance, double[] coefficients) {
        double innerProduct = coefficients[0];
	    for (int i = 0; i < currentInstance.numAttributes(); i++) {
            if (i != currentInstance.classIndex()) {
                innerProduct += coefficients[i + 1] * currentInstance.value(i);
            }
        }
        return innerProduct;
    }

    /**
	 * Returns the prediction of a linear regression predictor with weights
	 * given by m_coefficients on a single instance.
     *
	 * @param instance
	 * @return
	 * @throws Exception
	 */
	public double regressionPrediction(Instance instance) throws Exception {
		return instanceCoefficientsInnerProduct(instance, m_coefficients);
	}
	
	/**
	 * Calculates the total squared error over the data on a linear regression
	 * predictor with weights given by m_coefficients.
     *
	 * @param data
	 * @return
	 * @throws Exception
	 */
	public double calculateMSE(Instances data) throws Exception {
        Instance currentInstance;
        double currentCost = 0;
        for (int i = 0; i < data.numInstances(); i++) {
            currentInstance = data.instance(i);
            currentCost += Math.pow(regressionPrediction(currentInstance) - currentInstance.classValue(), 2);
        }

        return currentCost / (2 * data.numInstances());	}

    public double getAlpha() {
	    return m_alpha;
    }
    
    @Override
	public double classifyInstance(Instance arg0) throws Exception {
		// Don't change
		return 0;
	}

	@Override
	public double[] distributionForInstance(Instance arg0) throws Exception {
		// Don't change
		return null;
	}

	@Override
	public Capabilities getCapabilities() {
		// Don't change
		return null;
	}
}
