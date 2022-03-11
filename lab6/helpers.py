import holoviews as hv
import pandas as pd
hv.extension('bokeh')
import numpy as np
import panel as pn
import warnings
warnings.filterwarnings('ignore')

def guess_linear_regression_plot(m, c, N=100):
    np.random.seed(0)
    heights = np.random.normal(loc=66, scale=6, size=N)
    np.random.seed(1)
    weights_factors = np.random.normal(loc=1.86, scale=0.15, size=N)
    weights = heights * weights_factors
    dots = hv.Points(pd.DataFrame({'Height': heights, 'Weight': weights}), kdims=['Height', 'Weight']).opts(
        title='Height vs Weight',
        xlabel='Height (inches)',
        ylabel='Weight (lbs)',
        width=600,
        height=400,
        size=5
    )
    line = hv.Slope(m, c).opts(color='purple', line_width=2)
    return dots * line

def actual_linear_regression_plot(N=100):
    np.random.seed(0)
    heights = np.random.normal(loc=66, scale=6, size=N)
    np.random.seed(1)
    weights_factors = np.random.normal(loc=1.86, scale=0.15, size=N)
    weights = heights * weights_factors 

    #plot this dataset
    dots = hv.Points(pd.DataFrame({'Height': heights, 'Weight': weights}), kdims=['Height', 'Weight']).opts(
        title='Height vs Weight',
        xlabel='Height (inches)',
        ylabel='Weight (lbs)',
        width=600,
        height=400,
        size=5
    )
    # find the best fit line
    m, c = np.polyfit(heights, weights, 1)

    slope = hv.Slope(m, c).opts(color='red', line_width=2)
    if c < 0:
        print("Calculated best fit line: y = %.2fx %.2f" % (m, c))
    else:
        print("Calculated best fit line: y = %.2fx + %.2f" % (m, c))
    return dots * slope

def polynomial_fit_plot(degree, N=20):
    np.random.seed(12)
    xmax = 4
    x = np.linspace(0, xmax, N)
    y = np.cos(x) + 0.8*np.random.rand(N)
    p = np.poly1d(np.polyfit(x, y, degree))

    t = np.linspace(-0.5, xmax+0.5, 200)

    dots = hv.Points(pd.DataFrame({'x': x, 'y': y}), kdims=['x', "y"], vdims=['y']).opts(width=800, height=400, title='', xlabel='x', ylabel='y')
    # make the dots larger
    dots.opts(size=10)

    curve = hv.Curve(pd.DataFrame({'x': t, 'y': p(t)}), kdims=['x'], vdims=['y']).opts(color='red', line_width=2).redim(x=hv.Dimension('x', range=(-0.5, xmax+0.5)), y=hv.Dimension('y', range=(np.min(y)-0.5, np.max(y)+0.5)))

    print("Sum of squared errors: {:.4f}".format(np.sum((p(x) - y)**2)))
    return (dots * curve)