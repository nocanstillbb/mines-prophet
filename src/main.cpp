
#include "viewmodels/gameveiwmodel.h"

#include <prism/qt/modular/interfaces/intf_module.h>
#include <prism/qt/modular/wrapper.h>
#include <prism/container.hpp>
#include <prism/qt/core/helper/stopwatch.h>


#include <QQuickView>
#include <QGuiApplication>
#include <QQmlApplicationEngine>
#include <qqml.h>
#include <QPluginLoader>

void registerTypes();
prism::qt::modular::intfModule* loadplugin(const std::string& module_name);

int main(int argc, char* argv[])
{
#if QT_VERSION < QT_VERSION_CHECK(6, 0, 0)
    QCoreApplication::setAttribute(Qt::AA_EnableHighDpiScaling);
#endif

    QGuiApplication app(argc, argv);

    std::shared_ptr<QQmlApplicationEngine> sptr_engine(new QQmlApplicationEngine(),[](auto* p){ p->deleteLater(); }) ;

    prism::Container::get()->register_instance<QQmlApplicationEngine>(sptr_engine);


    prism::qt::modular::wrapper::startupUrl = "qrc:/mines-prophet/views/main.qml";

    registerTypes();
    // load plugins
    std::vector<prism::qt::modular::intfModule*> plugins;
    plugins.push_back(loadplugin("prism_qt_core"));
    plugins.push_back(loadplugin("prism_qt_ui"));
    prism::qt::modular::wrapper w(plugins, [&]() {
        static QMetaObject::Connection connection = QObject::connect(
            sptr_engine.get(), &QQmlApplicationEngine::objectCreated, &app, [&](QObject* object, const QUrl& url) {
                if (url.toString() == QString::fromStdString(prism::qt::modular::wrapper::startupUrl))
                {
                    if (!object)
                        app.exit(-1);
                    else
                    {
                    auto* win = reinterpret_cast<QQuickWindow*>(object);
                    if (win)
                    {
#ifdef _WIN32
                        win->setVisible(false);
                        win->setOpacity(0);
                        win->setWindowState(Qt::WindowMinimized);
#endif

                        //退出后释放opengl共享纹理
                        // "设置主窗口不释放纹理和场景图,退出后统一释放";
                        win->setPersistentOpenGLContext(true);
                        win->setPersistentSceneGraph(true);
                        std::shared_ptr<QQuickWindow> sp_win(win, [](QQuickWindow* p) { Q_UNUSED(p) });
                        prism::Container::get()->register_instance(sp_win);
                    }
                    if (!object)
                        app.exit(-1);
                    else
                        QObject::disconnect(connection);
                    }
                }
            },
            Qt::QueuedConnection);

        //项目窗口
        sptr_engine->load(QString::fromStdString(prism::qt::modular::wrapper::startupUrl));
        int result = app.exec();
        if (result)
            return result;

        return result;
    });

    int exitCode = w.run();
    STOPWATCHER_DUMP2FILE;
    return exitCode;

}

void registerTypes()
{
    qRegisterMetaType<QImage>("QImage");
    qRegisterMetaType<QRect>("QRect");
    qRegisterMetaType<QEventLoop*>("QEventLoop*");

    qRegisterMetaType<bool*>("bool*");
    qRegisterMetaType<double*>("double*");
    qRegisterMetaType<double>("double");
    qRegisterMetaType<float*>("float*");

    qRegisterMetaType<prism::qt::core::prismModelListProxy<CellInfo>*>("prismModelListProxy<CellInfo>*");
    qRegisterMetaType<prism::qt::core::prismModelProxy<CellInfo>*>("prismModelProxy<CellInfo>*");

    qmlRegisterSingletonInstance<GameVeiwmodel>("viewmodels", 1, 0, "GameVeiwmodel",new GameVeiwmodel());


}
prism::qt::modular::intfModule* loadplugin(const std::string& module_name)
{

    QString item = QString::fromStdString(module_name);
    QPluginLoader loader(item);
    QObject* plugin = loader.instance();
    if (!plugin)
        qDebug() << "plugin is null : " << loader.errorString();
    prism::qt::modular::intfModule* pi = qobject_cast<prism::qt::modular::intfModule*>(plugin);
    if (pi == nullptr)
        qDebug() << "pi is null";
    else
    {
        pi->setObjectName(item);
        return pi;
    }
    return (prism::qt::modular::intfModule*)nullptr;
}
