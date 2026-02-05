#pragma once

#include <cmath>
#include <vector>

using vec = double*;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifndef M_E
#define M_E 2.71828182845904523536
#endif

struct Func {
    virtual double operator()(const vec& x, int n) = 0;
    virtual void df(const vec& x, vec& g, int n) = 0;
};

//  Simple quadratic f(x) = x^2
struct Quadratic1D : Func {
    double operator()(const vec& x, int n) { return x[0] * x[0]; }
    void df(const vec& x, vec& g, int n) { g[0] = 2.0 * x[0]; }
};

//  2D quadratic bowl f(x,y) = x^2 + y^2
struct Quadratic2D : Func {
    double operator()(const vec& x, int n) { return x[0] * x[0] + x[1] * x[1]; }
    void df(const vec& x, vec& g, int n) {
        g[0] = 2.0 * x[0];
        g[1] = 2.0 * x[1];
    }
};

// Rosenbrock function f(x,y) = (1-x)^2 + 100(y-x^2)^2
struct Rosenbrock : Func {
    double operator()(const vec& x, int n) {
        double s = 0;
        for (int i = 0; i < n - 1; i++) {
            s +=
                100 * (x[i + 1] - x[i] * x[i]) * (x[i + 1] - x[i] * x[i]) + (1 - x[i]) * (1 - x[i]);
        }
        return s;
    }
    void df(const vec& x, vec& g, int n) {
        g[0] = -400 * x[0] * (x[1] - x[0] * x[0]) + 2 * (x[0] - 1);
        g[n - 1] = 200 * (x[n - 1] - x[n - 2] * x[n - 2]);
        for (int i = 1; i < n - 1; i++)
            g[i] = -400 * x[i] * (x[i + 1] - x[i] * x[i]) + 2 * (x[i] - 1)
                   + 200 * (x[i] - x[i - 1] * x[i - 1]);
    }
};

// Beale function f(x,y) = (1.5-x+xy)^2 + (2.25-x+xy^2)^2 + (2.625-x+xy^3)^2
struct Beale : Func {
    double operator()(const vec& x, int n) {
        double t1 = 1.5 - x[0] + x[0] * x[1];
        double t2 = 2.25 - x[0] + x[0] * x[1] * x[1];
        double t3 = 2.625 - x[0] + x[0] * x[1] * x[1] * x[1];
        return t1 * t1 + t2 * t2 + t3 * t3;
    }
    void df(const vec& x, vec& g, int n) {
        double t1 = 1.5 - x[0] + x[0] * x[1];
        double t2 = 2.25 - x[0] + x[0] * x[1] * x[1];
        double t3 = 2.625 - x[0] + x[0] * x[1] * x[1] * x[1];

        g[0] = 2.0 * t1 * (-1.0 + x[1]) + 2.0 * t2 * (-1.0 + x[1] * x[1])
               + 2.0 * t3 * (-1.0 + x[1] * x[1] * x[1]);
        g[1] = 2.0 * t1 * x[0] + 2.0 * t2 * 2.0 * x[0] * x[1] + 2.0 * t3 * 3.0 * x[0] * x[1] * x[1];
    }
};

// Himmelblau function: f(x,y) = (x^2 + y - 11)^2 + (x + y^2 - 7)^2
struct Himmelblau : Func {
    double operator()(const vec& x, int n) {
        double a = x[0] * x[0] + x[1] - 11.0;
        double b = x[0] + x[1] * x[1] - 7.0;
        return a * a + b * b;
    }
    void df(const vec& x, vec& g, int n) {
        double a = x[0] * x[0] + x[1] - 11.0;
        double b = x[0] + x[1] * x[1] - 7.0;
        g[0] = 4.0 * a * x[0] + 2.0 * b;
        g[1] = 2.0 * a + 4.0 * b * x[1];
    }
};

// Booth function: f(x,y) = (x + 2y - 7)^2 + (2x + y - 5)^2
struct Booth : Func {
    double operator()(const vec& x, int n) {
        double a = x[0] + 2.0 * x[1] - 7.0;
        double b = 2.0 * x[0] + x[1] - 5.0;
        return a * a + b * b;
    }
    void df(const vec& x, vec& g, int n) {
        double a = x[0] + 2.0 * x[1] - 7.0;
        double b = 2.0 * x[0] + x[1] - 5.0;
        g[0] = 2.0 * (a + 2.0 * b);
        g[1] = 2.0 * (2.0 * a + b);
    }
};

// Matyas function: f(x,y) = 0.26*(x^2 + y^2) - 0.48*x*y
struct Matyas : Func {
    double operator()(const vec& x, int n) {
        return 0.26 * (x[0] * x[0] + x[1] * x[1]) - 0.48 * x[0] * x[1];
    }
    void df(const vec& x, vec& g, int n) {
        g[0] = 0.52 * x[0] - 0.48 * x[1];
        g[1] = 0.52 * x[1] - 0.48 * x[0];
    }
};

// McCormick function: f(x,y) = sin(x + y) + (x - y)^2 - 1.5x + 2.5y + 1
struct McCormick : Func {
    double operator()(const vec& x, int n) {
        return std::sin(x[0] + x[1]) + (x[0] - x[1]) * (x[0] - x[1]) - 1.5 * x[0] + 2.5 * x[1]
               + 1.0;
    }
    void df(const vec& x, vec& g, int n) {
        g[0] = std::cos(x[0] + x[1]) + 2.0 * (x[0] - x[1]) - 1.5;
        g[1] = std::cos(x[0] + x[1]) - 2.0 * (x[0] - x[1]) + 2.5;
    }
};

struct Rastrigin : Func {
    double operator()(const vec& x, int n) const {
        double s = 10.0 * n;
        for (int i = 0; i < n; i++) s += x[i] * x[i] - 10.0 * std::cos(2.0 * M_PI * x[i]);
        return s;
    }

    void df(const vec& x, vec& g, int n) const {
        for (int i = 0; i < n; i++) g[i] = 2.0 * x[i] + 20.0 * M_PI * std::sin(2.0 * M_PI * x[i]);
    }
};

struct Ackley : Func {
    double operator()(const vec& x, int n) const {
        double sum1 = 0.0, sum2 = 0.0;
        for (int i = 0; i < n; i++) {
            sum1 += x[i] * x[i];
            sum2 += std::cos(2.0 * M_PI * x[i]);
        }
        double term1 = -20.0 * std::exp(-0.2 * std::sqrt(sum1 / n));
        double term2 = -std::exp(sum2 / n);
        return term1 + term2 + 20.0 + M_E;
    }

    void df(const vec& x, vec& g, int n) const {
        double sum1 = 0.0, sum2 = 0.0;
        for (int i = 0; i < n; i++) {
            sum1 += x[i] * x[i];
            sum2 += std::cos(2.0 * M_PI * x[i]);
        }
        double sqrt_sum1 = std::sqrt(sum1 / n);
        double exp1 = std::exp(-0.2 * sqrt_sum1);
        double exp2 = std::exp(sum2 / n);
        for (int i = 0; i < n; i++) {
            double grad1 =
                (sqrt_sum1 > 1e-12) ? (0.2 * 2.0 * x[i] / (n * sqrt_sum1) * 20.0 * exp1) : 0.0;
            double grad2 = (2.0 * M_PI / n) * std::sin(2.0 * M_PI * x[i]) * exp2;
            g[i] = grad1 + grad2;
        }
    }
};