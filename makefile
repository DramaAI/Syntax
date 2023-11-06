build:
	bash ./bash/build.sh $(FLAG)

help:
	@make build FLAG=-h

requirement:
	@make build FLAG=-r

rm-syntax:
	@rm -r ./syntax