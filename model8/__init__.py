from pyramid.config import Configurator


def main(global_config, **settings):
    """ This function returns a Pyramid WSGI application.
    """
    config = Configurator(settings=settings)
    config.include("cornice")
    # config.include('pyramid_chameleon')
    config.include('.models')
    config.scan(".views")
    return config.make_wsgi_app()
