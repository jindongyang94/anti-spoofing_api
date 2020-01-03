.PHONY: init build test

## Display file structures and content
tree:
	tree -I '*.png|*.jpg|*.pyc|*.mov|*.mp4|*pycache*|*.doctree|*.rst*|*.js|*.html|*.css'

## Stop and remove container image from local repo to rebuild
kill:
	bash scripts/run.sh kill

## Build image and deploy locally in docker
build:
	bash scripts/run.sh build

## Test by running two images into the system
test:
	bash scripts/run.sh test

## Clean the local repo to free up space
clean:
	docker system prune -a --volumes
	docker image prune -a

## Rerun the process when the docker does not work, by stopping, cleaning and rebuilding + redeploying
rerun: kill clean build