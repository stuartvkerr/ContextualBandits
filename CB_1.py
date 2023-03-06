import numpy as np
import plotly.offline
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from scipy.optimize import minimize
# NOTE: Ensure to import `cuffinks` python package (did so via pycharm package manager


class UserGenerator(object):
    """ Class to create synthetic data by simulating player dynamics. """

    def __init__(self):
        """
        Here we fix each offer's logistic function beta (weight) parameters,
        which the casino operator will need to learn. The number
        of beta parameters corresponds to the number of elements in the context vector.
        Later on we will calculate these weights via Bayesian methods.
        """
        self.beta = {}
        self.beta['A'] = np.array([-4, -0.1, -3, 0.1])
        self.beta['B'] = np.array([-6, -0.1, 1, 0.1])
        self.beta['C'] = np.array([2, 0.1, 1, -0.1])
        self.beta['D'] = np.array([4, 0.1, -3, -0.2])
        self.beta['E'] = np.array([-0.1, 0, 0.5, -0.1])
        self.context = None

    def logistic(self, beta: list, context: list) -> float:
        """
        Calculates the probability of an offer acceptance when an offer is
        displayed to the player

        :param beta: list of an offer's logistic function beta parameters
        :param context: context vector
        :return: probability as determined by logistic function

        """
        f = np.dot(beta, context)
        p = 1. / (1 + np.exp(-f))
        return p

    def display_offer(self, offer):
        """
        The CB model will use the context to display one of the 5 offers.
        The chosen offer will be passed to this function, producing a reward
        of a 1 or 0.

        Generates a reward (acceptance) with probability p determined by logistic function.

        :param offer: designation for offer
        :return: reward (0 or 1) determined by binomial distribution using `p`

        """
        if offer in ['A', 'B', 'C', 'D', 'E']:
            p = self.logistic(self.beta[offer], self.context)
            reward = np.random.binomial(n=1, p=p)
            return reward
        else:
            raise Exception('Unknown offer')

    def generate_user_with_context(self) -> list:
        """
        Generates a player and associated context vector.

        :return: context vector to be used in Logistic Regression equation.

        """
        # 0: Unhosted, 1: Hosted
        hosted = np.random.binomial(n=1, p=0.5)
        # 0: Female, 1: Male
        gender = np.random.binomial(n=1, p=0.75)
        # Generate player age between 18 and 77, with mean age 42
        # Note that mean of Beta function is = a/(a+b)
        # Not to confuse this beta function with beta values for Logistic Regression
        age = 18 + int(np.random.beta(2, 3) * 60)
        # add 1 to the context for the intercept in Logistic Regression equation.
        self.context = [1, gender, hosted, age]
        return self.context


def get_scatter(x, y, name, showlegend) -> object:
    dashmap = {'A': 'solid',
               'B': 'dot',
               'C': 'dash',
               'D': 'dashdot',
               'E': 'longdash'}
    s = go.Scatter(x=x, y=y, legendgroup=name, showlegend=showlegend, name=name,
                   line=dict(color='blue', dash=dashmap[name]))
    return s


def visualize_offers(ug):
    offer_list = 'ABCDE'
    ages = np.linspace(18, 77)
    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=("Female, Unhosted",
                                        "Female, Hosted",
                                        "Male, Unhosted",
                                        "Male, Hosted"))
    for gender in [0, 1]:
        for hosted in [0, 1]:
            showlegend = (gender == 0) & (hosted == 0)
            for offer in offer_list:
                probs = [ug.logistic(ug.beta[offer],
                                     [1, gender, hosted, age]) for age in ages]
                fig.add_trace(get_scatter(ages, probs, offer, showlegend),
                              row=gender + 1,
                              col=hosted + 1)

    # fig.update_layout(template = "presentation")
    fig.show()


if __name__ == '__main__':
    ug = UserGenerator()
    visualize_offers(ug)
