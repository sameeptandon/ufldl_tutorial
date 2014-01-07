---
layout: post
title:  "Authoring Content"
date:   2013-08-28 13:32:12
categories: authoring, development
---

This document explains how to setup your development environment to author
content.

## Preliminaries ## 
We use [Jekyll](http://jekyllrb.com) to build the site and git to deploy the site. The site is currently hosted by Github Pages. Changes pushed to Github typically take 10 minutes to show up. Hence, it is best to build the site locally, make all your changes, and then push to git when you are done.  

So how exactly do you build the site locally? 

1. Download the [github repo](http://github.com/sameeptandon/ufldl_tutorial). Navigate to the gh-pages of the branch.

2. Install [Jekyll via Github Pages](https://help.github.com/articles/using-jekyll-with-pages):
        
        apt-get install ruby1.9.1-dev
        gem install bundler
        bundle install

The `bundle install` command should be done at the root of the repository (in the gh-pages branch) where the `Gemfile` is. 

3. Build and run the site via

        bundle exec jekyll serve -w --baseurl ''

4. In your browser, navigate to [http://localhost:4000](http://localhost:4000)

That is it! Any changes you make to content will now be automatically updated to your local site.  

## Writing Content ## 

All content is written in Markdown (.md) files. This [cheat sheet](http://support.mashery.com/docs/customizing_your_portal/Markdown_Cheat_Sheet) goes over the syntax for the files. This will tell you how to make headers, paragraphs, lists, links, insert pictures, etc. 

You will also need to define some Jekyll specific options. At the beginning of each file, you must provide the following header (change title, date, and categories):

    ---
    layout: post
    title:  "Authoring Content"
    date:   2013-08-28 13:32:12
    categories: authoring, development
    ---

### Code blocks ### 

Syntax Highlighting with code blocks is possible. Check out the file about.md. Here's an example of `C++` code.

{% highlight cpp %}
int main(void) {
  printf("Hello world!");
  return 0;
}
{% endhighlight %}

### Latex Support ###

Latex support is obtained by using the `<m>` and `</m>` tags. For example, the code `<m> x = 0 </m>` will render <m> x = 0 </m>. Display type equations are also possible; just puts the tags on their own line. (See author.md in the repository for an example).  

<m>
\begin{aligned} 
\dot{x} &amp;= \sigma(y-x) \\
\dot{y} &amp;= \rho x - y - xz \\
\dot{z} &amp;= -\beta z + xy
\end{aligned}
</m>


<m>
\frac{1}{\Bigl(\sqrt{\phi \sqrt{5}}-\phi\Bigr) e^{\frac25 \pi}} =
1+\frac{e^{-2\pi}} {1+\frac{e^{-4\pi}} {1+\frac{e^{-6\pi}}
{1+\frac{e^{-8\pi}} {1+\ldots} } } }
</m>

Rendering is handled by MathJax. 
