# fastai-coursenlp-v2_rewrite
Attempt to rewriting fastai course nlp from https://github.com/fastai/course-nlp with fastai v2. 

For more information, please refer to the original repository including following blog post and lecture videos. 

### Further note: 
- These rewrites does not necessarily means it's the best way to do. One wasn't the most familiar with fastai API and there might be better way to do some. In any case, fell free to post in issues/discussion forum if you have a better idea of implementation. 
- Due to lack of skills, some code are unable to rewrite as one doesn't fully understand them as well, or how to get them back. 
- There might be some functions that acts differently than the original document. One did not fully test this out, and treated it works "as is". Ensure you are aware of this drawbacks and perhaps find your own implementation of ideas as well. 

If there are any other notices such as lacking in references, please post in issues/discussion to notify and one will try to find it back. Otherwise you could provide me some links and one will add it into the documents. Or you could open pull request with descriptions on what changes you'd made if they're not self-explaining. 

Other thing to note is the requirement is using fastai v2.4 (or 2.4.1). Older version of fastai are not tried. However, newer version (2.5) might have some changes. Example, some file have `Config.config_path` as their path variable. However, this no longer works in v2.5.1 as the code for `Config` no longer acquires the fastai path for you. Instead, you requires manually define the path. 

Spacy also requires v2 rather than v3 as they works differently. 

Thanks. 
