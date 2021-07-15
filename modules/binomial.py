import numpy as np
import scipy.stats as stats
import pandas as pd

import os

import plotly.graph_objects as go

class BinomialExperiment():
    """
    Creates an object that represents observed or desired split test results.
    Currently only supports two-way split tests. n-way tests will be supported
    in a future release.

    Analyses can then be performed on this object by calling this class's methods.
    Return statistical power, estimate a necessary sample size, return statistical
    significance. Plotting is also possible.

    Marketing program/campaign optimizaiton is the intended use case. Therefore,
    split tests are evaluated with one-way significance tests and under these hypotheses:

    Null: Treatment Probability - Control Probability <= 0
    Alt: Treatment Probability - Control Probability > 0

    Also, this class is designed to be used as the backend of a web application
    that helps marketers plan and understand optimization experiments.
    """
    def __init__(self, p_control = 0, p_treatment = 0, n_control = 0, n_treatment = 0, power = None, alpha = 0.05):
        """
        Only two required args are p_control and p_treatment. It is assumed that the user is either evaluating a completed
        experiment or has already determined the practical difference necessary to make an experiment's results worthwhile.

        So, those two values are already on-hand.
        """
        self.p_control = p_control
        self.p_treatment = p_treatment

        self.n_control = n_control
        self.n_treatment = n_treatment

        self.var_control = 1 * p_control * (1 - p_control)
        self.var_treatment = 1 * p_treatment * (1 - p_treatment)

        self.norm_null = None
        self.norm_alt = None

        self.binom_null = None
        self.binom_alt = None

        self.binom_control = None
        self.binom_treatment = None

        self.confidence_control = None
        self.confidence_treatment = None

        if n_control > 0 and n_treatment > 0 and p_control > 0 and p_treatment > 0:
            control = self.p_control * self.n_control
            treatment = self.p_treatment * self.n_treatment
            sample = self.n_control + self.n_treatment

            self.p_sample = (control + treatment) / sample
        else:
            self.p_sample = None

        if power == 1:
            print('Sample size approaches infinity as power approaches 1, so 1 is an invalid power vlaue. Changing power to 0.99.')
            self.power = 0.99
        elif power == 0:
            print('Sample size is undefined at power of 0, so 0 is an invalid power value. Changing power to 0.01.')
            self.power = 0.01
        else:
            self.power = power

        self.alpha = alpha
        self.p_value = None

    def get_p_sample(self):
        """
        Take sample sizes and probabilities from each sample and return the probability of the combination of samples
        """
        control = self.p_control * self.n_control
        treatment = self.p_treatment * self.n_treatment
        sample = self.n_control + self.n_treatment

        p_sample = (control + treatment) / sample

        self.p_sample = p_sample

        return p_sample

    def estimate_sample(self, power = None, alpha = None):
        """
        Take desired effect size, alpha and desired power level from self. Return a minimum sample size (one group)
        that would be necessary to acheive the desired experiment results.

        Allows the user to specify power here, if they didn't specify them when they instantiated the class.
        Otherwise, it takes the values provided to the class on instantiation.

        NOTE: If a power value is supplied, this method will NOT change self.power. The ability
        to set a power value here is designed to enable what-if testing scenarios. In those cases,
        a user would not be changing experiment parameters but rather quickly checking to see what
        would happen to sample size if power changed.
        """
        if power == None:
            power = self.power
        elif power > 0 and power < 1:
            power = power
        else:
            raise ValueError('Power provided is impossible (1, 0 or negative). Please provide a positive power between 0 and 1.')

        if alpha == None:
            alpha = self.alpha
        elif alpha > 0 and alpha < 1:
            alpha = alpha
        else:
            raise ValueError('Alpha provided is impossible (1, 0 or negative). Please provide a positive power between 0 and 1.')

        z_null = stats.norm.ppf(1 - self.alpha)
        z_alt = stats.norm.ppf(1 - power)

        stdev_null = np.sqrt(self.var_control + self.var_control)
        stdev_alt = np.sqrt(self.var_control + self.var_treatment)

        z_diff = (z_null * stdev_null) - (z_alt * stdev_alt)
        p_diff = self.p_treatment - self.p_control

        n = (z_diff / p_diff) ** 2

        sample_size = int(np.ceil(n))

        # Don't update self.power if this was just a what-if simulation.
        # Only update self.power if this is run to update experiment parameters.
        if power == self.power:
            self.n_control = sample_size
            self.n_treatment = sample_size

        return sample_size

    def binom_distribution(self):
        """
        Simulates two binomial distributions, one for control group and other
        for treatment group. Stored as attributes of the object created by this class.

        Allows a user to specify the probability of each distribution. Assuming separate
        distributions by default to keep things intuitive.

        But, when testing experiment results against a typical marketing hypothesis
        (null: treatment - control <= 0), both p_control and p_treatment should be
        set to the overall probability of the combined sample (call get_p_sample()
        method of this class, first). That way, a single null distribution can be
        generated to represent the difference between control and treatment being 0.

        (See .simulate_significance() for an example application like the above).
        """
        null_control = stats.binom.rvs(p = self.p_sample, n = self.n_control, size = 1000000) / self.n_control
        null_treatment = stats.binom.rvs(p = self.p_sample, n = self.n_treatment, size = 1000000) / self.n_treatment

        alt_control = stats.binom.rvs(p = self.p_control, n = self.n_control, size = 1000000) / self.n_control
        alt_treatment = stats.binom.rvs(p = self.p_treatment, n = self.n_treatment, size = 1000000) / self.n_treatment

        self.binom_null = null_treatment - null_control
        self.binom_alt = alt_treatment - alt_control

        self.binom_control = alt_control
        self.binom_treatment = alt_treatment

    def norm_distribution(self):
        """
        Approximate null and alt binomial distributions by simulating normal
        distributions. Normal distributions are created like this:

        Null: Treatment P = Control P, so combined distribution is control - control.
        Mean is 0, because null hypothesis is that treatment - control <= 0.

        Alt: Treatment P > Control P, so combined distribution is treatment - control.
        Mean is p_treatment - p_control, because alt hypothesis is treatment - control > 0.

        This function is useful when simulating statistical power post-hoc.
        """
        # Null hypothesis is no difference between treatment and control distributions
        # So, null distribution is control subtracted from itself (treatment = control)
        sterror_null = np.sqrt((self.var_control / self.n_control) + (self.var_control / self.n_control))
        # Alt hypothesis is treatment - control > 0
        # So, alt distribution is treatment - control, variance of which is var(treatment) + var(control).
        sterror_alt =  np.sqrt((self.var_treatment / self.n_treatment) + (self.var_control / self.n_control))

        self.sterror_null = sterror_null
        self.sterror_alt = sterror_alt

        dist_null = stats.norm(loc = 0, scale = sterror_null)
        dist_alt = stats.norm(loc = self.p_treatment - self.p_control, scale = sterror_alt)
        self.norm_null = dist_null
        self.norm_alt = dist_alt

    def confidence_intervals(self, level = 95):
        """
        Calculate level% confidence intervals for control distribution and treatment
        distribution. If plot == True, also return a plot contrasting the two
        intervals.

        Useful insight in addition to p value and power to understand how confident
        we can be in an experiment's conclusion (contrast interval overlap).
        """
        margin = (100 - level) / 2 # interval is middle level% of vals, so this is margin to either side of it
        try:
            len(self.binom_control)
            len(self.binom_treatment)

        except:
            self.binom_distribution()

        control = self.binom_control
        treatment = self.binom_treatment

        control_upper = np.percentile(a = control, q = level + margin)
        control_lower = np.percentile(a = control, q = margin)
        self.interval_control = {'lower': control_lower, 'upper':control_upper, 'level':level}

        treatment_upper = np.percentile(a = treatment, q = level + margin)
        treatment_lower = np.percentile(a = treatment, q = margin)
        self.interval_treatment = {'lower': treatment_lower, 'upper':treatment_upper, 'level':level}

        return self.interval_control, self.interval_treatment

    def plot_confidence(self, level = None, show = False):
        """
        Looks for confidence intervals in self.interval_control and self.interval_treatment.
        Plots them together for easy contrast.
        Returns a fig and will call plotly's fig.show() if show == True.

        If level is provided, method will calculate new confidence intervals with the provided level.
        If not and confidence intervals have already been calculated, level used during previous calc will be used.
        If not and this is the first time confidence intervals are being calculated, 95% will be assumed.
        """
        try: # Check to see if intervals have been calculated, already
            if level and level != self.interval_control['level']:
                self.confidence_intervals(level = level)
            else:
                level = self.interval_control['level']
        except: # If not, calculate them
            if level == None:
                level = 95
            self.confidence_intervals(level = level)

        int_control = [self.interval_control[k] for k in ['lower','upper']]
        int_treatment = [self.interval_treatment[k] for k in ['lower','upper']]

        low_end = min(int_control[0], int_treatment[0])
        high_end = max(int_control[1], int_treatment[1])
        r = high_end - low_end
        low_lim = low_end - (0.1 * r)
        high_lim = high_end + (0.1 * r)

        data = [
            go.Scatter(
                mode = 'lines+markers',
                line = dict(color = 'blue', width = 4),
                marker = dict(color = 'black', size = 10, symbol = 'line-ns-open'),
                x = int_control,
                y = [0.75 for i in range(len(int_control))],
                name = 'Control'
            ),
            go.Scatter(
                mode = 'lines+markers',
                line = dict(color = 'orange', width = 4),
                marker = dict(color = 'black', size = 10, symbol = 'line-ns-open'),
                x = int_treatment,
                y = [1.25 for i in range(len(int_treatment))],
                name = 'Treatment'
            )
        ]

        layout = dict(
            title = '{}% Confidence Intervals, Treatment vs Control'.format(level),
            plot_bgcolor = 'white',
            height = 350,
            width = 800,
            xaxis = dict(title = 'Probabilities',
                            range = (low_lim, high_lim),
                            showgrid = False,
                            zeroline = False,
                            showline = True,
                            linecolor = 'black',
                            tickformat = ',.0%'),
            yaxis = dict(range = (0,2),
                            showgrid = False,
                            zeroline = False,
                            showline = True,
                            linecolor = 'black',
                            visible = False)
        )

        fig = go.Figure(data = data, layout = layout)

        if show:
            fig.show()

        return fig

    def analyze_significance(self):
        """
        Take sample sizes and probabilities and return the significance of the difference between the probabilities.
        One-tailed test.

        Null: Treatment Prob - Control Prob <= 0
        Alt: Treatment Prob - Control Prob > 0
        """
        var_control = 1 * self.p_sample * (1 - self.p_sample)
        var_treatment = 1 * self.p_sample * (1 - self.p_sample) # Same as var_control, because null hyp is no difference

        sigma = np.sqrt((var_control / self.n_control) + (var_treatment / self.n_treatment))

        z = (self.p_treatment - self.p_control) / sigma
        p = (1 - stats.norm.cdf(z))
        self.p_value = p

        return p

    def simulate_significance(self):
        """
        Same intent and outcome as analyze_significance(), but it simulates a binomial distribution rather than
        approximating one with a normal distribution. No continuity correction, necessary. Only significant source of
        inaccuracy would be variability between runs (random simulations can yield slightly different outcomes, each time).
        """
        observed_difference = self.p_treatment - self.p_control

        try: # check to see if there's an array in self.binom_null
            len(self.binom_null)
            differences = self.binom_null
        except:
            self.binom_distribution()
            differences = self.binom_null

        p = (differences >= observed_difference).mean()
        self.p_value = p

        return p

    def simulate_power(self):
        """
        Takes results of a completed experiment and reveals the statistical power of the significance conclusion.
        """
        if self.p_treatment - self.p_control < 0:
            thresh = 1 - self.alpha
        else:
            thresh = self.alpha

        try:
            p_crit = self.norm_null.ppf(1 - thresh)
            beta = self.norm_alt.cdf(p_crit)
        except:
            self.norm_distribution()
            p_crit = self.norm_null.ppf(1 - thresh)
            beta = self.norm_alt.cdf(p_crit)

        power = (1 - beta) if self.p_treatment > self.p_control else beta
        self.power = power

        return power

    def plot_p(self, show = False):
        """
        Plot the null distribution, treatment probability and then shade the p value in order to visualize the results
        of a significance test.
        """
        try:
            difference = self.binom_null
        except:
            self.simulate_significance()
            difference = self.binom_null

        observed_difference = self.p_treatment - self.p_control

        mu, sigma = stats.norm.fit(difference)
        crit_density = stats.norm.pdf(observed_difference, mu, sigma)

        x = np.linspace(min(difference), max(difference), self.n_control + self.n_treatment)
        y = stats.norm.pdf(x, mu, sigma)

        line_curve = dict(color = 'blue', width = 2)

        data = [
            go.Scatter(
                x = x,
                y = y,
                mode = 'lines',
                showlegend = False,
                line = line_curve
            ),
            go.Scatter(
                x = x[x > observed_difference],
                y = y[np.where(x > observed_difference)],
                fill = 'tozeroy',
                showlegend = False,
                line = line_curve
            )
        ]

        layout = dict(
            plot_bgcolor = 'white',
            width = 800,
            height = 600,
            title = 'Significance',
            xaxis = dict(
                title = 'Difference in Probabilities',
                showgrid = False,
                zeroline = False,
                showline = True,
                linecolor = 'black'
            ),
            yaxis = dict(
                title = 'Density',
                showgrid = False,
                zeroline = False,
                showline = True,
                linecolor = 'black'
            )
        )

        fig = go.Figure(data = data, layout = layout)

        fig.add_vline(x = observed_difference,
                        line_width = 2,
                        line_dash = 'dash',
                        line_color = 'black',
                        annotation_text = 'P Value {:.4f}'.format(self.p_value),
                        annotation_position = 'top right')

        if show:
            # Intended to be used in notebooks.
            # .py app files that use this module will handle saving and opening from desktop
            fig.show();

        return fig

    def plot_power(self, show = False):
        """
        Produce a plot demonstrating the statistical power of the binomial split
        test's results.

        Plots a simulated null distribution, a simulated alt distribution, the
        critical value of the null distribution.

        Null p and alt beta are shaded to convey power in shorthand.

        Needs a value in self.power and self.norm_null to work. Call .simulate_power() before this
        to populate power, sim_null and sim_alt attributes.
        """
        if self.p_treatment - self.p_control < 0:
            thresh = 1 - self.alpha
        else:
            thresh = self.alpha

        try:
            p_crit = self.norm_null.ppf(1 - thresh)
            beta = self.norm_alt.cdf(p_crit)
        except:
            self.simulate_power()
            p_crit = self.norm_null.ppf(1 - thresh)
            beta = self.norm_alt.cdf(p_crit)

        sample_null = self.norm_null.rvs(size = self.n_control)
        sample_alt = self.norm_alt.rvs(size = self.n_treatment)

        lowest_x = min(min(sample_null), min(sample_alt))
        highest_x = max(max(sample_null), max(sample_alt))

        x = np.linspace(lowest_x, highest_x, self.n_control + self.n_treatment)

        y_null = self.norm_null.pdf(x)
        y_alt = self.norm_alt.pdf(x)

        # Set line parameters for visual styling
        line_null = dict(color = 'blue', width = 2)
        line_alt = dict(color = 'orange', width = 2)

        # Plot the null and alt distributions
        data = [
            go.Scatter(
                x = x,
                y = y_null,
                mode = 'lines',
                name = 'Null',
                line = line_null
            ),
            go.Scatter(
                x = x,
                y = y_alt,
                mode = 'lines',
                name = 'alt',
                line = line_alt
            ),
            # Shade P under null distribution
            go.Scatter(
                x = x[x > p_crit],
                y = y_null[np.where(x > p_crit)],
                fill = 'tozeroy',
                showlegend = False,
                line = line_null
            ),
            # Shade beta under alt distribution
            go.Scatter(
                x = x[x < p_crit],
                y = y_alt[np.where(x < p_crit)],
                fill = 'tozeroy',
                showlegend = False,
                line = line_alt
            )
        ]

        # Apply axis configurations to the plot
        layout = dict(
            yaxis = dict(
                showgrid = False,
                title = 'Probability Density',
                showline = True,
                linecolor = 'black',
                zeroline = False
            ),
            xaxis = dict(
                showgrid = False,
                title = 'Sample Mean Diffrences (Probabilities)',
                showline = True,
                linecolor = 'black',
                zeroline = False
            ),
            plot_bgcolor = 'white',
            width = 800,
            height = 600,
            title = 'Power'
        )

        fig = go.Figure(data = data, layout = layout)

        # Mark p_crit with a dashed vertical line
        fig.add_vline(x = p_crit,
                    line_width = 2,
                    line_dash = 'dash',
                    line_color = 'black',
                    annotation_text = 'P Crit (Power {:.2f})'.format(self.power),
                    annotation_position = 'top right')

        if show:
            fig.show()

        return fig

    def plot_power_curve(self, show = False):
        """
        Creates a line plot that shows how power changes as sample size changes.
        Intended to be used during experiment planning in order to find out
        if feasibility will be an issue.

        For instance, if the standard 0.80 power level requires a sample size
        that will take too long to generate, how much smaller can we go before
        power becomes a prohibitive issue?

        Requires effect size (p_treatment and p_control) and alpha to work. Then,
        it loops through many different power values and plots resulting sample
        size for each.
        """
        power_levels = np.linspace(0.01,0.99,176)
        sample_sizes = []

        for p in power_levels:
            size = self.estimate_sample(power = p)
            sample_sizes.append(size)

        x = power_levels
        y = sample_sizes

        line_curve = dict(color = 'blue', width = 2)

        x_axis = dict(title = 'Statistical Power', showline = True, linecolor = 'black', zeroline = False, showgrid = False)
        y_axis = dict(title = 'Recommended Size per Sample at Alpha 0.05', showline = True, linecolor = 'black', zeroline = False, showgrid = False, tickformat = ',d')

        fig = go.Figure()
        fig.add_trace(go.Scatter(x = x, y = y, mode = 'lines', showlegend = False, line = line_curve))
        fig.add_vline(x = self.power, line_dash = 'dash', line_color = 'black', line_width = 2)

        fig.update_xaxes(x_axis)
        fig.update_yaxes(y_axis)
        fig.update_layout(plot_bgcolor = 'white',
                            width = 800,
                            height = 600,
                            title = 'Sample Curve')

        if show:
            fig.show();

        return fig

    def evaluate(self, plot = False, show = False, summary = True):
        """
        Calls other methods in this class in order to speed up the experiment evaluation
        process and make this class more intuitive to use.

        User can treat this class as a container for parameters of an experiment that has
        concluded. Calling evaluate on it will generate P, Power and some Plots if plot == True.

        Will call plt.show(); on each plot, if show == True.
        """
        self.get_p_sample()
        self.simulate_significance()
        self.simulate_power()
        self.confidence_intervals()

        if summary:
            print(self)

        if plot:
            fig1 = self.plot_p(show = show)
            fig2 = self.plot_power(show = show)
            fig3 = self.plot_confidence(show = show)

            return fig1, fig2, fig3

    def plan(self, plot = False, show = False, summary = True):
        """
        Call other methods in this class in order to speed up the experiment planning
        flow and make this class more intuitive to use.

        When planning an experiment, user will leave n_control and n_treatment blank.
        This is because the objective there is figuring out how many observations to generate
        in order to detect a minimum practical effect size at a reasonable degree of power.

        Before calling this method, user has populated p_control, p_treatment, power and alpha.
        Power is desired power level. p_control is status quo rate. p_treatment is minimum outcome
        rate required to be meaningful to the business. Alpha is desired significance level
        (almost always, 0.05 is desired).
        """
        self.estimate_sample()
        self.n_control
        self.get_p_sample()
        self.simulate_significance()
        self.confidence_intervals()

        if summary:
            print(self)

        if plot:
            fig1 = self.plot_p(show = show)
            fig2 = self.plot_power(show = show)
            fig3 = self.plot_power_curve(show = show)
            fig4 = self.plot_confidence(show = show)

            return fig1, fig2, fig3, fig4

    def __repr__(self):
        """
        Magic method that outputs the experiment's parameters, so far.
        """
        header = '|||Experiment Readout|||\n'
        data = [['Control Probability', '{:.2%}'.format(self.p_control)],
               ['Treatment Probability', '{:.2%}'.format(self.p_treatment)],
               ['Effect Size', '{:.2%}'.format(self.p_treatment - self.p_control)],
               ['',''],
               ['Control Sample Size', '{:,}'.format(self.n_control)],
               ['Treatment Sample Size', '{:,}'.format(self.n_treatment)],
               ['',''],
               ['Statistical Power', '{:.3f}'.format(self.power) if self.power else 'None'],
               ['Significance Threshold', '{:.3f}'.format(self.alpha)],
               ['P Value', '{:.3f}'.format(self.p_value) if self.p_value != None else 'None']]

        return header + str(pd.DataFrame(data = [x[1] for x in data], index = [x[0] for x in data], columns = ['']))
